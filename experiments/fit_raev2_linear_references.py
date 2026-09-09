"""Exact weighted affine reference fits on a frozen native feature map."""
import copy,csv,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as n
from experiments.train_raev2_observable_potential import load_banks,batch_from_bank

@torch.no_grad()
def main():
 out=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_reference_linear_fit_20260908');out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 banks,bank_record=load_banks(Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1'))
 n.install_raev2_decoder_config_compat();cfg=n.load_config(n.DEFAULT_CONFIG);model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
 ck=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect();assert model.s_patch_size==1
 versions=[p._version for p in model.parameters()];captured={}
 hook=model.base_final_layer.linear.register_forward_pre_hook(lambda mod,args:captured.update(phi=args[0].detach()))
 outer_hook=model.base_final_layer.register_forward_pre_hook(lambda mod,args:captured.update(feature=args[0].detach()))
 def draw(bank,idx,gen,drop):
  x,y=batch_from_bank(bank,idx,'cuda');raw=torch.randn(len(idx),generator=gen,device='cuda').sigmoid();t=8*raw/(1+7*raw);eps=torch.randn(x.shape,generator=gen,device='cuda');z=(1-t[:,None,None,None])*x+t[:,None,None,None]*eps
  if drop:y=torch.where(torch.rand(len(idx),generator=gen,device='cuda')<.1,torch.full_like(y,1000),y)
  return x,y,t,z
 def forward(z,t,y):
  with torch.autocast('cuda',dtype=torch.bfloat16):f,b=model(z,t,context=y,attn_mask=None)
  return f,b,captured.pop('phi'),captured.pop('feature')
 layer=model.base_final_layer.linear;d=layer.in_features+1;c=layer.out_features
 theta0=torch.cat([layer.weight.T,layer.bias[None,:]]).to(torch.bfloat16).double()
 G=torch.zeros(d,d,device='cuda',dtype=torch.float64);B={k:torch.zeros(d,c,device='cuda',dtype=torch.float64) for k in ['data','teacher']};energy={k:torch.zeros((),device='cuda',dtype=torch.float64) for k in B}
 gen=torch.Generator(device='cuda').manual_seed(202609421);started=time.perf_counter();tokens=0
 for start in range(0,5000,16):
  idx=np.arange(start,min(start+16,5000));x,y,t,z=draw(banks['train'],idx,gen,True);f,b,phi,feat=forward(z,t,y)
  phi=phi.to(torch.bfloat16).double();phi=torch.cat([phi,torch.ones(*phi.shape[:-1],1,device='cuda',dtype=torch.float64)],-1)
  weights=t.double().clamp_min(.05)[:,None,None];p=(phi/weights).flatten(0,1);G.add_(p.T@p);tokens+=len(p)
  for name,target in [('data',x),('teacher',f)]:
   v=(target.double().flatten(2).transpose(1,2)/weights).flatten(0,1);B[name].add_(p.T@v);energy[name].add_((v*v).sum())
  if start==0 or (start+len(idx))%1000==0:print(json.dumps({'cached':start+len(idx),'seconds':time.perf_counter()-started}),flush=True)
 eig,Q=torch.linalg.eigh((G+G.T)/2);tol=torch.finfo(torch.float64).eps*max(tokens,d)*eig[-1];assert eig[0]>-tol
 keep=eig>tol;inv=torch.where(keep,1/eig.clamp_min(float(tol)),torch.zeros_like(eig))
 def solve(b):return Q@(inv[:,None]*(Q.T@b))
 theta={k:solve(b) for k,b in B.items()};direct=solve(B['teacher']-B['data']);difference=theta['teacher']-theta['data'];linearity_error=float((direct-difference).norm()/direct.norm().clamp_min(1e-20));assert linearity_error<1e-8
 heads={k:copy.deepcopy(model.base_final_layer) for k in B};fit=[]
 def risk(a,k):return float((energy[k]-2*(a*B[k]).sum()+(a*(G@a)).sum())/(tokens*c))
 for name,h in heads.items():
  h.linear.weight.copy_(theta[name][:-1].T.float());h.linear.bias.copy_(theta[name][-1].float())
  quant=theta[name].to(torch.bfloat16).double()
  residual=float((G@theta[name]-B[name]).norm()/B[name].norm())
  fit.append(dict(head=name,original_risk=risk(theta0,name),fitted_fp64_risk=risk(theta[name],name),fitted_bf16_coeff_risk=risk(quant,name),normal_equation_relative_residual=residual))
  assert fit[-1]['fitted_fp64_risk']<=fit[-1]['original_risk']+1e-7
 fit_seconds=time.perf_counter()-started
 gen=torch.Generator(device='cuda').manual_seed(202609420);rows=[]
 for start in range(0,1000,16):
  idx=np.arange(start,min(start+16,1000));x,y,t,z=draw(banks['validation'],idx,gen,False);f,b,phi,feat=forward(z,t,y)
  with torch.autocast('cuda',dtype=torch.bfloat16):pred={k:model.unpatchify(h(feat,feat),1) for k,h in heads.items()}
  pred['original']=b;den=t[:,None,None,None].clamp_min(.05)
  for name,value in pred.items():
   data=((value.float()-x)/den).square().flatten(1).mean(1).cpu().tolist();teacher=((value.float()-f.float())/den).square().flatten(1).mean(1).cpu().tolist()
   for j in range(len(idx)):rows.append(dict(head=name,sample=int(idx[j]),noise_time=float(t[j]),data_velocity_mse=data[j],teacher_velocity_mse=teacher[j]))
 assert versions==[p._version for p in model.parameters()]
 for name,h in heads.items():torch.save({'state_dict':h.state_dict(),'target':name,'fit':'weighted_affine_pseudoinverse'},out/f'{name}.pt')
 for filename,values in [('validation.csv',rows),('fit.csv',fit)]:
  with (out/filename).open('x') as file:
   w=csv.DictWriter(file,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 hook.remove();outer_hook.remove();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 meta=dict(complete=True,source_parameters_unchanged=True,train_samples=5000,validation_samples=1000,train_tokens=tokens,feature_dim=d,rank=int(keep.sum()),lambda_min=float(eig[0]),lambda_max=float(eig[-1]),numerical_threshold=float(tol),condition_retained=float(eig[-1]/eig[keep][0]),linearity_relative_error=linearity_error,fit_seconds=fit_seconds,seconds=time.perf_counter()-started,bank=bank_record,source_checkpoint_sha256=sha(n.DEFAULT_CHECKPOINT),source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'docs/RAEV2_REFERENCE_LINEAR_FIT_PROTOCOL_20260908_ZH.md'))
 (out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'complete.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
