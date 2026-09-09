"""Verify source preservation and heldout affine-reference results."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.sample_raev2_pfr_retiming import DEFAULT_CHECKPOINT

def main():
 r=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_reference_linear_fit_20260908');m=json.loads((r/'complete.json').read_text());assert m['complete'] and m['source_parameters_unchanged']
 assert m['rank']==m['feature_dim']==1441 and m['train_tokens']==1280000
 assert m['linearity_relative_error']<1e-8 and m['lambda_min']>m['numerical_threshold']
 assert hashlib.sha256((r/'source.py').read_bytes()).hexdigest()==m['source_sha256']
 ck=torch.load(DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)['ema']
 for name in ['data','teacher']:
  p=torch.load(r/f'{name}.pt',map_location='cpu',weights_only=True);assert p['target']==name
  for k,v in p['state_dict'].items():
   assert torch.isfinite(v).all()
   if not k.startswith('linear.'):assert torch.equal(v,ck['base_final_layer.'+k])
 d=pd.read_csv(r/'validation.csv');assert len(d)==3000
 old=pd.read_csv('/home/zhoushunyu/data/eqvae/experiments/raev2_reference_refit_20260908/validation_0000.csv')
 cols=['sample','noise_time','data_velocity_mse','teacher_velocity_mse']
 assert d[d['head']=='original'][cols].reset_index(drop=True).equals(old[old['head']=='original'][cols].reset_index(drop=True))
 fit=pd.read_csv(r/'fit.csv');assert (fit.normal_equation_relative_residual<1e-10).all()
 assert (fit.fitted_fp64_risk<=fit.original_risk).all()
 baseline=d[d['head']=='original'].sort_values('sample');rows=[]
 for name in ['original','data','teacher']:
  x=d[d['head']==name].sort_values('sample');assert np.array_equal(x['sample'],np.arange(1000))
  row={'head':name,'samples':len(x)}
  for k in ['data_velocity_mse','teacher_velocity_mse']:
   a=x[k].to_numpy();delta=a-baseline[k].to_numpy();assert np.isfinite(a).all()
   row[k]=a.mean();row[k+'_delta_original']=delta.mean();row[k+'_paired_se']=delta.std(ddof=1)/len(a)**.5
  rows.append(row)
 pd.DataFrame(rows).to_csv('experiments/results/terminal_defect_20260908/raev2_linear_reference_summary.csv',index=False,mode='x')
 fit.to_csv('experiments/results/terminal_defect_20260908/raev2_linear_reference_fit.csv',index=False,mode='x')
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
