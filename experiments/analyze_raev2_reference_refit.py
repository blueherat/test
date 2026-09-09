"""Audit fixed reference refits; report predictive and response changes, not quality."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def main():
 root=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_reference_refit_20260908');m=json.loads((root/'complete.json').read_text())
 assert all(m[k] for k in ['complete','source_parameters_unchanged','source_validation_bitwise_unchanged','initial_heads_bitwise_parity'])
 assert m['steps']==2048 and m['batch_size']==16 and m['trainable_parameters_per_head']==5627104
 assert hashlib.sha256((root/'source.py').read_bytes()).hexdigest()==m['source_sha256']
 hist=pd.read_csv(root/'training.csv');assert np.array_equal(hist.step,np.arange(1,2049)) and np.isfinite(hist.to_numpy()).all()
 before=pd.read_csv(root/'validation_0000.csv');after=pd.read_csv(root/'validation_2048.csv')
 cols=['sample','noise_time','response_active','data_velocity_mse','teacher_velocity_mse','response_rms','response_cos_full','response_mse_to_full']
 assert len(before)==len(after)==3000
 original=before[before['head']=='original'][cols].reset_index(drop=True)
 assert original.equals(after[after['head']=='original'][cols].reset_index(drop=True))
 for h in ['data','teacher']:
  assert original.equals(before[before['head']==h][cols].reset_index(drop=True))
  c=torch.load(root/f'{h}_2048.pt',map_location='cpu',weights_only=False)
  assert c['steps']==2048 and c['target']==h
  assert sum(v.numel() for v in c['state_dict'].values())==5627104
  assert all(torch.isfinite(v).all() for v in c['state_dict'].values())
  assert all(float(v['step'])==2048 for v in c['optimizer']['state'].values())
 rows=[]
 for head in ['original','data','teacher']:
  d=after[after['head']==head].sort_values('sample');assert np.array_equal(d['sample'],np.arange(1000))
  assert np.array_equal(d.noise_time,original.noise_time)
  for region,selected in [('all',np.ones(len(d),dtype=bool)),('noise_gt_half',d.response_active.to_numpy())]:
   x=d[selected];b=original[selected];row=dict(head=head,region=region,samples=len(x))
   for k in ['data_velocity_mse','teacher_velocity_mse','response_rms','response_cos_full','response_mse_to_full']:
    v=x[k].to_numpy();delta=v-b[k].to_numpy();assert np.isfinite(v).all()
    row[k]=v.mean();row[k+'_delta_original']=delta.mean();row[k+'_paired_se']=delta.std(ddof=1)/np.sqrt(len(delta))
   rows.append(row)
 out=pd.DataFrame(rows);out.to_csv('experiments/results/terminal_defect_20260908/raev2_reference_refit_summary.csv',index=False,mode='x')
 print(out[['head','region','samples','data_velocity_mse','teacher_velocity_mse','response_rms','response_cos_full','response_mse_to_full']].to_string(index=False))
 print(json.dumps({k:m[k] for k in ['train_seconds','seconds','trainable_parameters_per_head']}))
if __name__=='__main__':main()
