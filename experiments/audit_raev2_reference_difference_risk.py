"""Fixed unit-coefficient reference difference: heldout norm and error alignment."""
import copy,csv,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as n
from experiments.train_raev2_observable_potential import load_banks,batch_from_bank

@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 root=Path('/home/zhoushunyu/data/eqvae/experiments');out=root/'raev2_reference_difference_risk_20260908';out.mkdir(exist_ok=False)
 paths={'one':root/'raev2_reference_linear_fit_20260908','ten':root/'raev2_reference_multinoise_20260908'}
 banks,bank_record=load_banks(root/'raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1')
 n.install_raev2_decoder_config_compat();cfg=n.load_config(n.DEFAULT_CONFIG);model=n.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
 ck=torch.load(n.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect()
 versions=[p._version for p in model.parameters()];heads={};old={};hashes={}
 for arm,path in paths.items():
  old[arm]=pd.read_csv(path/'validation.csv').set_index(['head','sample'])
  for name in ['data','teacher']:
   h=copy.deepcopy(model.base_final_layer);h.load_state_dict(torch.load(path/f'{name}.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True);heads[arm,name]=h
   hashes[f'{arm}_{name}']=hashlib.sha256((path/f'{name}.pt').read_bytes()).hexdigest()
 captured={};hook=model.base_final_layer.register_forward_pre_hook(lambda mod,args:captured.update(feature=args[0].detach()))
 gen=torch.Generator(device='cuda').manual_seed(202609420);rows=[];started=time.perf_counter();parity_max=0.;identity_max=0.
 for start in range(0,1000,16):
  idx=np.arange(start,min(start+16,1000));x,y=batch_from_bank(banks['validation'],idx,'cuda');raw=torch.randn(len(idx),generator=gen,device='cuda').sigmoid();t=8*raw/(1+7*raw);eps=torch.randn(x.shape,generator=gen,device='cuda');z=(1-t[:,None,None,None])*x+t[:,None,None,None]*eps
  with torch.autocast('cuda',dtype=torch.bfloat16):
   f,b=model(z,t,context=y,attn_mask=None);feat=captured.pop('feature');pred={key:model.unpatchify(h(feat,feat),1) for key,h in heads.items()}
  den=t[:,None,None,None].clamp_min(.05);error=(f.float()-x)/den
  norm=lambda v:v.square().flatten(1).mean(1)
  for arm in paths:
   for name,v in [('original',b),('data',pred[arm,'data']),('teacher',pred[arm,'teacher'])]:
    mse=norm((v.float()-x)/den).cpu().numpy();expected=old[arm].loc[(name,idx),'data_velocity_mse'].to_numpy()
    parity_max=max(parity_max,float(np.max(np.abs(mse-expected))))
   d=(pred[arm,'teacher'].float()-pred[arm,'data'].float())/den
   inner=(error*d).flatten(1).mean(1);dn=norm(d);en=norm(error);direct=norm(error-d)-en;delta=dn-2*inner
   identity_max=max(identity_max,float((direct-delta).abs().max()))
   vals={k:v.cpu().tolist() for k,v in dict(error_norm2=en,difference_norm2=dn,error_inner_product=inner,unit_correction_risk_delta=delta,direct_risk_delta=direct).items()}
   for j,i in enumerate(idx):rows.append(dict(arm=arm,sample=int(i),noise_time=float(t[j]),**{k:v[j] for k,v in vals.items()}))
 assert parity_max<1e-7,parity_max
 assert identity_max<1e-6,identity_max
 assert versions==[p._version for p in model.parameters()];hook.remove();torch.cuda.synchronize();seconds=time.perf_counter()-started
 frame=pd.DataFrame(rows);frame.to_csv(out/'per_sample.csv',index=False);summary=[]
 for arm,df in frame.groupby('arm'):
  row={'arm':arm,'samples':len(df)}
  for k in ['error_norm2','difference_norm2','error_inner_product','unit_correction_risk_delta']:
   row[k]=df[k].mean();row[k+'_descriptive_se']=df[k].std(ddof=1)/len(df)**.5
  row['fraction_unit_improves']=float((df.unit_correction_risk_delta<0).mean());summary.append(row)
 portable=ROOT/'experiments/results/terminal_defect_20260908';pd.DataFrame(summary).to_csv(portable/'raev2_reference_difference_risk.csv',index=False,mode='x')
 stats=torch.load(paths['ten']/'sufficient_statistics.pt',map_location='cpu',weights_only=True);theta=stats['theta']['teacher']-stats['theta']['data'];scale=stats['tokens']*stats['output_dim']
 train_norm=float((theta*(stats['G']@theta)).sum()/scale);train_inner=float((theta*(stats['B']['teacher']-stats['B']['data'])).sum()/scale);assert abs(train_norm-train_inner)<1e-10
 meta=dict(complete=True,seconds=seconds,source_parameters_unchanged=True,validation_reproduction_max_error=parity_max,risk_identity_max_error=identity_max,train_fp64_difference_norm2=train_norm,train_fp64_error_inner_product=train_inner,bank=bank_record,head_hashes=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'complete.json').write_text(json.dumps(meta,indent=2));print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(meta))
if __name__=='__main__':main()
