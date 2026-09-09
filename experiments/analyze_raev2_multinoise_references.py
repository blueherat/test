"""Verify source preservation and heldout affine-reference results."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.sample_raev2_pfr_retiming import DEFAULT_CHECKPOINT

def main():
 r=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_reference_multinoise_20260908');m=json.loads((r/'complete.json').read_text());assert m['complete'] and m['source_parameters_unchanged']
 assert m['rank']==m['feature_dim']==1441 and m['train_tokens']==12800000 and m['noise_passes']==10 and m['train_noised_states']==50000
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
 stats=torch.load(r/'sufficient_statistics.pt',map_location='cpu',weights_only=True)
 G=stats['G'];assert stats['tokens']==m['train_tokens']
 chol=torch.linalg.cholesky((G+G.T)/2)
 audit=[]
 for name in ['data','teacher']:
  rhs=stats['B'][name];a=torch.cholesky_solve(rhs,chol);saved=stats['theta'][name]
  relative=float((a-saved).norm()/a.norm());assert relative<1e-9
  residual=float((G@a-rhs).norm()/rhs.norm());assert residual<1e-10
  risk=lambda v:float((stats['energy'][name]-2*(v*rhs).sum()+(v*(G@v)).sum())/(stats['tokens']*stats['output_dim']))
  row=fit[fit['head']==name].iloc[0]
  for key,coef in [('original_risk',stats['theta0']),('fitted_fp64_risk',a),('fitted_bf16_coeff_risk',saved.to(torch.bfloat16).double())]:
   assert abs(risk(coef)-float(row[key]))<1e-8,(name,key,risk(coef),row[key])
  audit.append(dict(head=name,cpu_cholesky_solution_relative_difference=relative,cpu_normal_equation_relative_residual=residual))
 pd.DataFrame(audit).to_csv('experiments/results/terminal_defect_20260908/raev2_multinoise_reference_cpu_audit.csv',index=False,mode='x')
 baseline=d[d['head']=='original'].sort_values('sample');rows=[]
 for name in ['original','data','teacher']:
  x=d[d['head']==name].sort_values('sample');assert np.array_equal(x['sample'],np.arange(1000))
  row={'head':name,'samples':len(x)}
  for k in ['data_velocity_mse','teacher_velocity_mse']:
   a=x[k].to_numpy();delta=a-baseline[k].to_numpy();assert np.isfinite(a).all()
   row[k]=a.mean();row[k+'_delta_original']=delta.mean();row[k+'_paired_se']=delta.std(ddof=1)/len(a)**.5
  rows.append(row)
 pd.DataFrame(rows).to_csv('experiments/results/terminal_defect_20260908/raev2_multinoise_reference_summary.csv',index=False,mode='x')
 fit.to_csv('experiments/results/terminal_defect_20260908/raev2_multinoise_reference_fit.csv',index=False,mode='x')
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
