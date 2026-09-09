"""Single-depth JiT IG tuning, then PFR and lifting; immutable per-arm requests."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments import jit_internal_guidance as jig
from experiments.raev2_capacity_lifting import heun_flow
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/jit_fine_sweep_20260909')
BASE=ROOT.parent/'jit_transfer_20260908'
TRAIN=ROOT.parent/'jit_internal_readouts_20260908'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True,choices=range(4));p.add_argument('--arm',required=True);p.add_argument('--depth',type=int,choices=[4],required=True);p.add_argument('--samples',type=int,default=1000);p.add_argument('--seed',type=int,default=202609831);p.add_argument('--method',choices=['ig','pfr','lifting'],required=True);p.add_argument('--early',type=float,required=True);p.add_argument('--late',type=float,required=True);p.add_argument('--iterations',type=int,default=1,choices=[1,2]);a=p.parse_args()
 assert json.loads((TRAIN/'complete.json').read_text())['complete']
 torch.cuda.set_device(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 model=jig.load_source('cuda');heads=jig.Readouts().cuda().eval().requires_grad_(False)
 state=torch.load(TRAIN/'last.pt',map_location='cpu',weights_only=False);assert state['step']==50000;heads.load_state_dict(state['ema']);del state
 out=ROOT/a.arm/f'rank{a.rank}';out.mkdir(parents=True,exist_ok=True)
 paths=[Path(__file__),Path(jig.__file__),jig.REPO/'model_jit.py',jig.REPO/'util/model_util.py',Path('experiments/raev2_capacity_lifting.py'),Path('docs/JIT_FINE_SWEEP_20260909_ZH.md')]
 req=dict(**vars(a),batch=4,steps=100,precision='FP32 state BF16 TF32',checkpoint_sha256=sha(jig.CHECKPOINT),head_sha256=sha(TRAIN/'last.pt'),sources={str(p):sha(p) for p in paths})
 if (out/'request.json').exists():assert json.loads((out/'request.json').read_text())==req
 else:atomic(out/'request.json',req)
 rh=sha(out/'request.json');counts={'full':0,'prefix':0};labels=None
 def pair(z,t):
  ts=t.expand(len(z));feats,c=jig.features(model,z,ts,labels,depths=(a.depth,12))
  full=jig.unpatchify(model.final_layer(feats['12'],c));weak=jig.unpatchify(heads.layers[str(a.depth)](feats[str(a.depth)],c));counts['full']+=1
  return jig.velocity(full,z,ts),jig.velocity(weak,z,ts)
 def field(z,t,kind):
  ts=t.expand(len(z))
  if kind=='base':
   f,c=jig.features(model,z,ts,labels,depths=(a.depth,));clean=jig.unpatchify(heads.layers[str(a.depth)](f[str(a.depth)],c));counts['prefix']+=1
  else:clean=model(z,ts,labels);counts['full']+=1
  return jig.velocity(clean,z,ts)
 grid=torch.linspace(0,1,101,device='cuda');rng=torch.Generator(device='cuda').manual_seed(a.seed);nh=hashlib.sha256();lh=hashlib.sha256();files=[]
 with torch.autocast('cuda',dtype=torch.bfloat16):
  for start in range(0,a.samples,4):
   z=torch.randn(4,3,256,256,device='cuda',generator=rng);labels=torch.arange(start,start+4,device='cuda')%1000;raw=z.cpu().numpy().tobytes();lab=labels.cpu().numpy().tobytes();nh.update(raw);lh.update(lab)
   if start//4%4!=a.rank:continue
   f=out/f'batch{start:04d}.npz'
   if not f.exists():
    if start==0:
     for t in [grid[0],grid[37],grid[80]]:
      vf,vw=pair(z,t);assert torch.equal(vf,field(z,t,'full'));assert torch.equal(vw,field(z,t,'base'))
     if a.seed==202609831:
      zz=z.clone()
      for t,u in zip(grid[:-1],grid[1:]):zz=zz+(u-t)*field(zz,t,'full')
      pp=np.round(np.clip(((zz+1)/2).cpu().numpy().transpose(0,2,3,1)*255,0,255)).astype(np.uint8)
      with np.load(BASE/'full_euler100/rank0/batch0000.npz') as old:np.testing.assert_array_equal(pp,old['arr_0'])
    before=counts.copy();torch.cuda.synchronize();beg=time.perf_counter();k=0
    while k<100:
     t=grid[k];gamma=a.early if k<50 else a.late
     if a.method=='lifting' and gamma:
      end=min(k+4,50 if k<50 else 100);sub=grid[k:end+1]
      for _ in range(a.iterations):
       target=heun_flow(z,sub,'full',field);lift=heun_flow(target,sub.flip(0),'base',field);z=z+(gamma/a.iterations)*(lift-z)
      for tt,u in zip(sub[:-1],sub[1:]):z=z+(u-tt)*field(z,tt,'full')
      k=end;continue
     if gamma:
      full,weak=pair(z,t);v=full+gamma*(full-weak)
      if a.method=='pfr' and k<50:
       h=min(1/32,.5-float(t));c=(1+gamma)*(full-weak)
       coef=(c*v).flatten(1).sum(1)/v.square().flatten(1).sum(1).clamp_min(1e-20)
       q=z+h*coef.clamp_min(0)[:,None,None,None]*v
       future=field(q,t+t.new_tensor(h),'base');v=v+(1+gamma)*(weak-future)
     else:v=field(z,t,'full')
     z=z+(grid[k+1]-t)*v;k+=1
    assert torch.isfinite(z).all();pixels=np.round(np.clip(((z+1)/2).cpu().numpy().transpose(0,2,3,1)*255,0,255)).astype(np.uint8);torch.cuda.synchronize()
    delta={k:counts[k]-before[k] for k in counts};tmp=f.with_suffix('.tmp')
    with tmp.open('wb') as fp:np.savez(fp,arr_0=pixels,labels=labels.cpu().numpy(),noise_sha256=hashlib.sha256(raw).hexdigest(),request_sha256=rh,seconds=time.perf_counter()-beg,full_calls=delta['full'],prefix_calls=delta['prefix'])
    tmp.replace(f)
   with np.load(f) as saved:assert str(saved['request_sha256'])==rh and str(saved['noise_sha256'])==hashlib.sha256(raw).hexdigest();np.testing.assert_array_equal(saved['labels'],labels.cpu().numpy())
   files.append(dict(start=start,file=f.name,sha256=sha(f)));atomic(out/'progress.json',dict(done=len(files)*4,arm=a.arm))
 ref=json.loads((BASE/'full_euler100/input_hashes.json').read_text())
 if a.samples==1000 and a.seed==202609831:assert nh.hexdigest()==ref['noise'] and lh.hexdigest()==ref['labels']
 atomic(out/'summary.json',dict(complete=True,samples=a.samples,seed=a.seed,files=files,noise_sha256=nh.hexdigest(),label_sha256=lh.hexdigest(),request_sha256=rh))
if __name__=='__main__':main()
