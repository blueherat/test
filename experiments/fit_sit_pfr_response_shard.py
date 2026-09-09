"""Amortize PFR's temporal weak response on new teacher trajectories, four shards."""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from experiments.run_internal_guidance_sit_audit import load_model
from experiments.audit_official_sit_pfr_interface import prefix
from experiments.raev2_training_core import file_sha256

ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/sit_pfr_response_distillation_20260908')
REPO=Path('research_repos/internal_guidance_study/Internal-Guidance/SiT').resolve()
CK=Path('/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt')

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=range(4),required=True);a=p.parse_args()
 out=ROOT/f'rank{a.rank}';out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 digest=file_sha256(CK);assert digest=='a7f4eb9f417295a14e5063b70020ab3f55b60a4905ccda7b244343deaf9840cd'
 model,meta=load_model(repo=REPO,checkpoint_path=CK,model_name='SiT-XL/2',encoder_depth=8,state_key='ema',device=torch.device('cuda'))
 model.eval().requires_grad_(False)
 request=dict(rank=a.rank,world_size=4,trajectories=1000,batch=4,seed=202609601,labels='0..999',teacher='PFR100 high-noise prefix',scale=1.35,horizon=1/32,fit='affine ridge, lambda=1e-4*trace(G)/d after merge',checkpoint_sha256=digest,sources={str(x):file_sha256(x) for x in [Path(__file__),Path('experiments/audit_official_sit_pfr_interface.py'),Path('experiments/run_internal_guidance_sit_audit.py'),REPO/'models/sit.py',Path('docs/SIT_PFR_RESPONSE_DISTILLATION_20260908_ZH.md')]})
 if (out/'request.json').exists(): assert json.loads((out/'request.json').read_text())==request
 else:(out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
 rh=file_sha256(out/'request.json');d=model.final_layer_xr.linear.in_features+1;c=model.final_layer_xr.linear.out_features
 G=torch.zeros(d,d,device='cuda',dtype=torch.float64);B=torch.zeros(d,c,device='cuda',dtype=torch.float64);energy=torch.zeros((),device='cuda',dtype=torch.float64)
 done=[];seconds=0.;tokens=0
 save=out/'statistics.pt'
 if save.exists():
  old=torch.load(save,map_location='cuda',weights_only=False);assert old['request_sha256']==rh
  G.copy_(old['G']);B.copy_(old['B']);energy.copy_(old['energy']);done=old['done'];seconds=old['seconds'];tokens=old['tokens']
 capture={}
 hook=model.final_layer_xr.linear.register_forward_hook(lambda mod,args,value:capture.update(phi=args[0],value=value))
 gen=torch.Generator(device='cuda').manual_seed(202609601);nh=hashlib.sha256();lh=hashlib.sha256();grid=torch.linspace(1,0,101,dtype=torch.float64)
 for start in range(0,1000,4):
  initial=torch.randn(4,4,32,32,generator=gen,device='cuda');labels=torch.arange(start,start+4,device='cuda')
  nh.update(initial.cpu().numpy().tobytes());lh.update(labels.cpu().numpy().tobytes())
  if (start//4)%4!=a.rank or start in done:continue
  begin=time.perf_counter();z=initial.double()
  for index in range(50):
   t,s=grid[index],grid[index+1];ts=torch.full((4,),float(t),device='cuda')
   full,base,_=model(z.float(),ts,labels);phi=capture['phi'];current=capture['value']
   if index==0:assert torch.equal(model.unpatchify(current),base)
   future=prefix(model,z.float(),torch.full_like(ts,max(.5,float(t)-1/32)),labels)
   target=(current-capture['value']).reshape(-1,c).float()
   X=phi.reshape(-1,d-1).float();X=torch.cat((X,torch.ones(len(X),1,device='cuda')),dim=1)
   G.add_((X.T@X).double());B.add_((X.T@target).double());energy.add_(target.double().square().sum());tokens+=len(X)
   drift=base+1.35*(full-base);drift=drift+1.35*(base-future)
   z=z+(s-t)*drift.double()
  assert torch.isfinite(z).all();torch.cuda.synchronize();seconds+=time.perf_counter()-begin;done.append(start)
  state=dict(G=G.cpu(),B=B.cpu(),energy=energy.cpu(),done=done,seconds=seconds,tokens=tokens,request_sha256=rh)
  temp=save.with_suffix('.tmp');torch.save(state,temp);temp.replace(save)
  if len(done)==1 or len(done)%10==0:print(json.dumps(dict(rank=a.rank,trajectories=len(done)*4,seconds=seconds)),flush=True)
 hook.remove()
 result=dict(complete=True,rank=a.rank,trajectories=len(done)*4,tokens=tokens,seconds=seconds,full_sample_calls=len(done)*4*50,prefix_sample_calls=len(done)*4*50,noise_sha256=nh.hexdigest(),label_sha256=lh.hexdigest(),statistics_sha256=file_sha256(save),request_sha256=rh)
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
