"""SiT IG counterpart of the frozen RAE native-path clock decomposition."""
import csv, hashlib, json, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import load_repo_modules,runtime_paths

@torch.inference_mode()
def main():
 out=Path('/home/zhoushunyu/data/eqvae/experiments/sit_clock_decomposition_20260908');out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 paths=runtime_paths(ROOT,Path('/home/zhoushunyu/data/eqvae/imagenet_sit_flow'),Path('/data/shared/envs/adm-fid/bin/python'));m=load_repo_modules(ROOT)
 sit,metadata=m['load_official_sit_module'](m['DEFAULT_OFFICIAL_SIT_REPO'],verify_source=True)
 model,semantics,_=m['load_sit_field_model'](checkpoint_path=paths['strong'],weights='ema',sit_module=sit,source_metadata=metadata,device=torch.device('cuda'))
 assert semantics.prediction_target=='velocity'
 head=m['load_internal_head_for_source'](checkpoint_path=paths['depth4'],name='depth4_v',head_weights='ema',model=model,sit_module=sit,source_checkpoint_path=paths['strong'],source_metadata=metadata,device=torch.device('cuda'))
 gen=torch.Generator(device='cuda').manual_seed(202609414);rows=[];calls=0;noise_hash=hashlib.sha256();started=time.perf_counter()
 def rms(x):return x.float().flatten(1).square().mean(1).sqrt()
 def cosine(x,y):
  x=x.float().flatten(1);y=y.float().flatten(1)
  return (x*y).sum(1)/(x.norm(dim=1)*y.norm(dim=1)).clamp_min(1e-20)
 for start in range(0,32,4):
  z=torch.randn(4,4,32,32,generator=gen,device='cuda');noise_hash.update(z.cpu().numpy().tobytes());labels=torch.arange(start,start+4,device='cuda')
  def fields(z,t):
   nonlocal calls
   full,heads,_=m['evaluate_source_with_heads'](model,z,torch.full((len(z),),t,device='cuda'),labels,heads={'depth4_v':head});calls+=1
   gamma=.6 if t<.25 else (.7 if t<.5 else 0.)
   gap=gamma*(full-heads['depth4_v']);return full+gap,full-gap
  for i in range(40):
   t=i/40;g,r=fields(z,t)
   if i in [0,5,15]:
    for H in [.025,.125]:
     q=z+.025*g;_,rt=fields(z,t+H);_,rq=fields(q,t+H)
     terms={'contrast':.025*(g-r),'clock':.025*(r-rt),'state':.025*(rt-rq)}
     total=sum(terms.values());direct=.025*(g-rq)
     assert torch.allclose(total,direct,atol=2e-6,rtol=2e-5)
     terms['total']=total;metrics={'native_rms':rms(.025*g),'latent_rms':rms(z),'identity_residual_rms':rms(total-direct)}
     for k,v in terms.items():
      metrics[k+'_rms']=rms(v);metrics[k+'_over_native']=rms(v)/rms(.025*g).clamp_min(1e-20);metrics[k+'_cos_native']=cosine(v,g)
     for x,y in [('contrast','clock'),('contrast','state'),('clock','state')]:metrics[x+'_cos_'+y]=cosine(terms[x],terms[y])
     metrics={k:v.cpu().tolist() for k,v in metrics.items()}
     for j in range(4):rows.append(dict(sample=start+j,index=i,noise_time=1-t,h=.025,H=H,**{k:v[j] for k,v in metrics.items()}))
   z=z+.025*g
  assert torch.isfinite(z).all()
  print(json.dumps({'done':start+4,'seconds':time.perf_counter()-started}),flush=True)
 with (out/'per_sample.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 (out/'complete.json').write_text(json.dumps(dict(complete=True,samples=32,seed=202609414,seconds=time.perf_counter()-started,batched_model_calls=calls,noise_sha256=noise_hash.hexdigest(),strong_sha256=sha(paths['strong']),head_sha256=sha(paths['depth4']),source_sha256=sha(__file__)),indent=2))

if __name__=='__main__':main()
