"""Conservative empirical mean-bias correction bounds for balanced 5/class banks.

No distribution-free confidence bound is claimed. Bounds apply to the usual
within-class plug-in bias correction over every balanced partition of features.
"""
import json,hashlib
from pathlib import Path
import torch
import numpy as np
R=Path(__file__).resolve().parents[1]/'experiments/results/terminal_defect_20260908'
def main():
 d=json.loads((R/'pfr_endpoint_moments.json').read_text());out=[]
 for group,control in [('sit5k','ordinary115'),('rae5k','ordinary')]:
  bounds={}
  for arm in [control,'pfr']:
   row=next(x for x in d['rows'] if x['group']==group and x['arm']==arm)
   p=Path(row['feature_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['feature_sha256']
   x=torch.load(p,map_location='cpu',weights_only=True).numpy().astype(float);assert x.shape==(5000,2048)
   x-=x.mean(0)
   # For C classes, k independent draws/class, plug-in tr Var(mean)
   # = sum_c sum_j ||x_cj - mean_c||^2 / [C^2 k(k-1)].
   # Every within-class sum of squares is <= total sum of squares.
   bounds[arm]=float(np.square(x).sum()/(1000**2*5*4))
  change=next(x['mean_term_change'] for x in d['changes'] if x['group']==group)
  row=dict(group=group,observed_mean_change=change,max_plugin_bias=bounds,
           corrected_change_lower=change-bounds['pfr'],corrected_change_upper=change+bounds[control])
  out.append(row);print(row,flush=True)
 with (R/'pfr_mean_bias_bound.json').open('x') as f:json.dump(dict(complete=True,rows=out,scope=__doc__),f,indent=2)
if __name__=='__main__':main()
