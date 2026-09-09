"""Create previously absent official JiT baselines, with shared input bank and four shards."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments import jit_internal_guidance as jig
from denoiser import Denoiser
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/jit_transfer_20260908')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=range(4),required=True);p.add_argument('--arm',choices=['full_euler100','cfg_heun50'],required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 model=jig.load_source('cuda');out=ROOT/a.arm/f'rank{a.rank}';out.mkdir(parents=True,exist_ok=True)
 request=dict(rank=a.rank,arm=a.arm,seed=202609831,batch=4,samples=1000,model='JiT-B/16',state='model_ema1',
  precision='FP32 state BF16 autocast TF32',cfg=3. if a.arm=='cfg_heun50' else 1.,cfg_interval=[.1,1.],
  checkpoint_sha256=sha(jig.CHECKPOINT),sources={str(p):sha(p) for p in [Path(__file__),Path(jig.__file__),jig.REPO/'model_jit.py',jig.REPO/'denoiser.py',jig.REPO/'util/model_util.py']})
 if (out/'request.json').exists():assert json.loads((out/'request.json').read_text())==request
 else:(out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
 rh=sha(out/'request.json');rng=torch.Generator(device='cuda').manual_seed(202609831);nh=hashlib.sha256();lh=hashlib.sha256();files=[];started=time.perf_counter()
 def field(z,t,y):
  times=torch.full((len(z),),float(t),device='cuda');f=model(z,times,y);v=jig.velocity(f,z,times)
  if a.arm=='full_euler100':return v
  b=model(z,times,torch.full_like(y,1000));vb=jig.velocity(b,z,times)
  scale=torch.where((times<1.)&(times>.1),3.,1.)[:,None,None,None]
  return vb+scale*(v-vb)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  for start in range(0,1000,4):
   z=torch.randn((4,3,256,256),generator=rng,device='cuda');labels=torch.arange(start,start+4,device='cuda')
   raw=z.cpu().numpy().tobytes();lab=labels.cpu().numpy().tobytes();nh.update(raw);lh.update(lab)
   if (start//4)%4!=a.rank:continue
   path=out/f'batch{start:04d}.npz'
   if path.exists():
    with np.load(path) as s:
     assert str(s['request_sha256'])==rh and str(s['noise_sha256'])==hashlib.sha256(raw).hexdigest()
     np.testing.assert_array_equal(s['labels'],np.arange(start,start+4))
   else:
    bs=time.perf_counter();steps=100 if a.arm=='full_euler100' else 50;grid=torch.linspace(0.,1.,steps+1,device='cuda')
    for k,(t,s) in enumerate(zip(grid[:-1],grid[1:])):
     v=field(z,t,labels);pred=z+(s-t)*v
     z=z+(s-t)*.5*(v+field(pred,s,labels)) if a.arm=='cfg_heun50' and k<steps-1 else pred
    assert torch.isfinite(z).all()
    parity=False
    if a.arm=='cfg_heun50' and start==0:
     official=Denoiser.__new__(Denoiser);torch.nn.Module.__init__(official);official.net=model
     official.noise_scale=1.;official.img_size=256;official.method='heun';official.steps=50
     official.num_classes=1000;official.cfg_scale=3.;official.cfg_interval=(.1,1.);official.t_eps=.05
     with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
      torch.cuda.manual_seed_all(202609831);reference=official.generate(labels)
     torch.testing.assert_close(z,reference,rtol=0,atol=0);parity=True
    pixels=np.round(np.clip(((z+1)/2).cpu().numpy().transpose(0,2,3,1)*255,0,255)).astype(np.uint8)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,arr_0=pixels,labels=np.arange(start,start+4),noise_sha256=hashlib.sha256(raw).hexdigest(),request_sha256=rh,seconds=time.perf_counter()-bs,official_latent_parity=parity)
    tmp.replace(path)
   files.append(dict(start=start,file=path.name,sha256=sha(path)))
   if len(files)==1 or len(files)%10==0:print(json.dumps(dict(arm=a.arm,rank=a.rank,samples=len(files)*4)),flush=True)
 result=dict(complete=True,rank=a.rank,samples=len(files)*4,files=files,noise_sha256=nh.hexdigest(),label_sha256=lh.hexdigest(),request_sha256=rh,seconds=time.perf_counter()-started)
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
