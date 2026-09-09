"""Paired four-rank ordinary IG with a Strong-only low-noise tail."""
import argparse,hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch
from experiments import sample_imagenet100_sit_foresight_fixed_point as s
from experiments.imagenet100_sit_multiscale_models import evaluate_internal_head_only
from experiments.raev2_capacity_lifting import advance_block
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/small_sit_two_stage_ig_20260909')
REF=ROOT.parent/'sit_pfr_output_control_20260908/v/ordinary'
DATA=Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow')
ARMS={'ig035_cut02':(.35,.8),'ig035_cut05':(.35,.5)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def segments(grid,alpha,end):
 # Integer step boundaries avoid float32 .8 being slightly larger than .8.
 cutoff=round(end*(len(grid)-1));k=0
 while k<len(grid)-1:
  stop=min(k+4,cutoff) if k<cutoff else k+1
  yield k,stop,alpha if k<cutoff else 0.
  k=stop
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=range(4),required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);device=torch.device('cuda');torch.cuda.set_device(0)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high')
 strong=DATA/'runs/sit-s-2_seed0/checkpoints/step_00800000.pt';head=DATA/'multiscale_guidance_study_v1/runs/depth4_v/checkpoints/step_00050000.pt'
 mod,src=s.load_official_sit_module(s.DEFAULT_OFFICIAL_SIT_REPO,verify_source=True)
 model,sem,meta,payload=s._load_field_model(checkpoint_path=strong,requested_field='auto',weights='ema',sit_module=mod,source_metadata=src,device=device);del payload
 spec=s.load_internal_head_for_source(checkpoint_path=head,name='depth4',head_weights='ema',model=model,sit_module=mod,source_checkpoint_path=strong,source_metadata=src,device=device)
 from diffusers.models import AutoencoderKL
 vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse',local_files_only=True).to(device).eval().requires_grad_(False)
 alpha,end=ARMS[a.arm];grid=torch.linspace(0,1,101,device=device);blocks=list(segments(grid,alpha,end));active=sum(j-i for i,j,c in blocks if c)
 out=ROOT/a.arm/f'rank{a.rank}';out.mkdir(parents=True,exist_ok=True)
 sources=[Path(__file__),Path(s.__file__),Path(inspect.getfile(advance_block)),Path(inspect.getfile(evaluate_internal_head_only)),Path(inspect.getfile(type(model))),Path('docs/SMALL_SIT_TWO_STAGE_IG_20260909_ZH.md')]
 request=dict(rank=a.rank,arm=a.arm,alpha=alpha,end_data_time=end,steps=100,batch=8,samples=1000,seed=202609417,precision='fp32',tf32=True,strong_sha256=sha(strong),head_sha256=sha(head),sources={str(q):sha(q) for q in sources},source_metadata=src,model_metadata=meta,reference_manifest_sha256=sha(REF/'sampling_manifest.json'),segments=blocks,method='ordinary shared-forward IG then Strong only',guided_steps=round(end*100))
 assert request['strong_sha256']=='b7f7d7318ee4b480fe591bc451c2aceb09efaf4111e57fd9dbefcd1bcfd88caa'
 assert request['head_sha256']=='f33fca87577ad2f5e70470f05c1834811a28e7912aedcc26a3fdb8ae8c4fc5ad'
 request=json.loads(json.dumps(request))
 if (out/'request.json').exists():assert json.loads((out/'request.json').read_text())==request
 else:atomic(out/'request.json',request)
 rh=sha(out/'request.json');counts={'full':0,'prefix':0};labels=None
 def field(z,t,kind):
  ts=t.expand(len(z))
  if kind=='base':counts['prefix']+=1;return evaluate_internal_head_only(model,z,ts,labels,spec=spec)
  counts['full']+=1
  return s._model_velocity(model,sem,z,t,labels,autocast_dtype=None)
 rng=torch.Generator(device=device).manual_seed(202609417);nh=hashlib.sha256();lh=hashlib.sha256();files=[]
 for start in range(0,1000,8):
  z=torch.randn((8,*s.LATENT_SHAPE),device=device,generator=rng);labels=torch.randint(0,s.NUM_CLASSES,(8,),device=device,generator=rng)
  nb=z.cpu().numpy().tobytes();lb=labels.cpu().numpy().tobytes();nh.update(nb);lh.update(lb)
  if start//8%4!=a.rank:continue
  f=out/f'batch{start:04d}.npz'
  if not f.exists():
   parity=None
   if start==0:
    # Reproduce only eight already-existing baseline images, for interface parity.
    zz=z.clone()
    for t,u in zip(grid[:-1],grid[1:]):
     full,weak,_=s.evaluate_source_with_heads(model,zz,t.expand(8),labels,heads={'depth4':spec},source_semantics=sem)
     if float(t)==0.:assert torch.equal(weak['depth4'],field(zz,t,'base'));assert torch.equal(full,field(zz,t,'full'))
     zz=zz+(u-t)*(full+.35*(full-weak['depth4']))
    pix=s.official_pixel_quantization(s.decode_latents_in_chunks(vae,zz,scaling_factor=s.SD_VAE_SCALING_FACTOR,chunk_size=2))
    with np.load(REF/'samples_n1000.npz') as saved:parity=np.array_equal(pix,saved['arr_0'][:8])
    assert parity,'native baseline pixel mismatch'
   before=counts.copy();torch.cuda.synchronize();beg=time.perf_counter()
   guided_calls=0
   for k,(t,u) in enumerate(zip(grid[:-1],grid[1:])):
    if k<round(end*100):
     full,weak,_=s.evaluate_source_with_heads(model,z,t.expand(8),labels,heads={'depth4':spec},source_semantics=sem)
     counts['full']+=1;guided_calls+=1
     drift=full+alpha*(full-weak['depth4'])
    else:drift=field(z,t,'full')
    z=z+(u-t)*drift
   assert guided_calls==round(end*100)
   assert torch.isfinite(z).all()
   pix=s.official_pixel_quantization(s.decode_latents_in_chunks(vae,z,scaling_factor=s.SD_VAE_SCALING_FACTOR,chunk_size=2));torch.cuda.synchronize()
   elapsed=time.perf_counter()-beg;delta={k:counts[k]-before[k] for k in counts};assert delta=={'full':100,'prefix':0}
   tmp=f.with_suffix('.tmp')
   with tmp.open('wb') as stream:np.savez(stream,arr_0=pix,labels=labels.cpu().numpy(),noise_sha256=hashlib.sha256(nb).hexdigest(),request_sha256=rh,seconds=elapsed,full_calls=delta['full'],prefix_calls=delta['prefix'],native_parity=str(parity),guided_shared_calls=guided_calls)
   tmp.replace(f)
  with np.load(f) as saved:
   assert str(saved['request_sha256'])==rh and str(saved['noise_sha256'])==hashlib.sha256(nb).hexdigest();np.testing.assert_array_equal(saved['labels'],labels.cpu().numpy())
  files.append(dict(start=start,file=f.name,sha256=sha(f)))
  atomic(out/'progress.json',dict(done=len(files)*8,arm=a.arm,rank=a.rank))
  if a.smoke:print(json.dumps(dict(smoke=True,file=str(f))),flush=True);return
 ref=json.loads((REF/'sampling_manifest.json').read_text())
 assert nh.hexdigest()==ref['noise_sha256'] and lh.hexdigest()==ref['label_sha256']
 atomic(out/'summary.json',dict(complete=True,rank=a.rank,arm=a.arm,files=files,noise_sha256=nh.hexdigest(),label_sha256=lh.hexdigest(),request_sha256=rh))
 print(a.arm,a.rank,'complete',flush=True)
if __name__=='__main__':main()
