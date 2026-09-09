"""Paired fixed-budget reference readouts: real targets versus frozen Full targets."""
import copy,csv,gc,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as n
from experiments.train_raev2_observable_potential import load_banks,batch_from_bank


def main():
 out=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_reference_refit_20260908');out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 banks,bank_record=load_banks(Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1'))
 n.install_raev2_decoder_config_compat();cfg=n.load_config(n.DEFAULT_CONFIG)
 model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
 ck=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect()
 versions=[p._version for p in model.parameters()];source_ids={id(p) for p in model.parameters()}
 heads={k:copy.deepcopy(model.base_final_layer).requires_grad_(True) for k in ['data','teacher']}
 assert not source_ids.intersection({id(p) for h in heads.values() for p in h.parameters()})
 opts={k:torch.optim.AdamW(h.parameters(),lr=1e-4,weight_decay=0.) for k,h in heads.items()}
 captured={}
 hook=model.base_final_layer.register_forward_pre_hook(lambda module,args:captured.update(feature=args[0].detach()))
 def forward(z,t,y):
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):full,base=model(z,t,context=y,attn_mask=None)
  return full,base,captured.pop('feature')
 def predict(h,feature):return model.unpatchify(h(feature,feature),model.s_patch_size)
 def norm(x):return x.float().flatten(1).square().mean(1).sqrt()
 def cosine(x,y):
  x=x.float().flatten(1);y=y.float().flatten(1)
  return (x*y).sum(1)/(x.norm(dim=1)*y.norm(dim=1)).clamp_min(1e-20)
 def draw(bank,idx,gen,drop):
  clean,labels=batch_from_bank(bank,idx,'cuda');raw=torch.randn(len(idx),generator=gen,device='cuda').sigmoid();t=8*raw/(1+7*raw)
  eps=torch.randn(clean.shape,generator=gen,device='cuda');z=(1-t[:,None,None,None])*clean+t[:,None,None,None]*eps
  if drop:labels=torch.where(torch.rand(len(idx),generator=gen,device='cuda')<.1,torch.full_like(labels,1000),labels)
  return clean,labels,t,z
 saved_source=None
 @torch.no_grad()
 def validate(stage):
  nonlocal saved_source
  gen=torch.Generator(device='cuda').manual_seed(202609420);rows=[];started=time.perf_counter()
  for start in range(0,1000,16):
   idx=np.arange(start,min(start+16,1000));clean,y,t,z=draw(banks['validation'],idx,gen,False)
   full,base,feat=forward(z,t,y)
   if start==0:
    if stage==0:saved_source=(full.clone(),base.clone())
    else:assert torch.equal(full,saved_source[0]) and torch.equal(base,saved_source[1])
   with torch.autocast('cuda',dtype=torch.bfloat16):preds={k:predict(h,feat) for k,h in heads.items()}
   if stage==0:
    for p in preds.values():assert torch.equal(p,base)
   preds['original']=base
   tf=(t-1/32).clamp_min(.05);ff,bf,featf=forward(z,tf,y)
   with torch.autocast('cuda',dtype=torch.bfloat16):future={k:predict(h,featf) for k,h in heads.items()}
   future['original']=bf
   den=t[:,None,None,None].clamp_min(.05);denf=tf[:,None,None,None]
   strong_delta=(z-full.float())/den-(z-ff.float())/denf
   for name,pred in preds.items():
    delta=(z-pred.float())/den-(z-future[name].float())/denf
    vals={'data_velocity_mse':((pred.float()-clean)/den).square().flatten(1).mean(1),'teacher_velocity_mse':((pred.float()-full.float())/den).square().flatten(1).mean(1),'response_rms':norm(delta),'response_cos_full':cosine(delta,strong_delta),'response_mse_to_full':(delta-strong_delta).square().flatten(1).mean(1)}
    vals={k:v.cpu().tolist() for k,v in vals.items()}
    for j in range(len(idx)):rows.append(dict(stage=stage,head=name,sample=int(idx[j]),noise_time=float(t[j]),response_active=bool(t[j]>.5),**{k:v[j] for k,v in vals.items()}))
  with (out/f'validation_{stage:04d}.csv').open('x') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print(json.dumps({'validation':stage,'seconds':time.perf_counter()-started}),flush=True)
 started=time.perf_counter();validate(0)
 gen=torch.Generator(device='cuda').manual_seed(202609419);rng=np.random.default_rng(202609418);history=[];train_start=time.perf_counter()
 for step in range(1,2049):
  clean,y,t,z=draw(banks['train'],rng.integers(0,5000,size=16),gen,True);full,base,feat=forward(z,t,y);den=t[:,None,None,None].clamp_min(.05)
  row={'step':step}
  for name,h in heads.items():
   opts[name].zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=predict(h,feat)
   target=clean if name=='data' else full.float();loss=((pred.float()-target)/den).square().mean();assert torch.isfinite(loss)
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(h.parameters(),1.,error_if_nonfinite=True);opts[name].step();row[name+'_loss']=float(loss);row[name+'_grad_norm']=float(grad)
  history.append(row)
  if step==16 or step%128==0:print(json.dumps({**row,'training_seconds':time.perf_counter()-train_start}),flush=True)
 train_seconds=time.perf_counter()-train_start
 assert versions==[p._version for p in model.parameters()] and all(p.grad is None for p in model.parameters())
 validate(2048)
 for name,h in heads.items():torch.save({'state_dict':h.state_dict(),'optimizer':opts[name].state_dict(),'steps':2048,'target':name},out/f'{name}_2048.pt')
 with (out/'training.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
 hook.remove()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 meta=dict(complete=True,steps=2048,batch_size=16,train_seconds=train_seconds,seconds=time.perf_counter()-started,source_parameters_unchanged=True,source_validation_bitwise_unchanged=True,initial_heads_bitwise_parity=True,trainable_parameters_per_head=sum(p.numel() for p in heads['data'].parameters()),source_checkpoint_sha256=sha(n.DEFAULT_CHECKPOINT),bank=bank_record,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'docs/RAEV2_REFERENCE_REFIT_PROTOCOL_20260908_ZH.md'))
 (out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'complete.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
