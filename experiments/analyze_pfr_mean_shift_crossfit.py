"""Fixed random-half validation of existing paired endpoint feature displacement.

Descriptive sample holdout, not unseen-class validation or a generation method.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1]/'experiments/results/terminal_defect_20260908'

def main():
 source=json.loads((R/'pfr_endpoint_moments.json').read_text());results=[]
 for group,control in [('sit5k','ordinary115'),('rae5k','ordinary')]:
  arrays=[];parents=[]
  for arm in [control,'pfr']:
   row=next(x for x in source['rows'] if x['group']==group and x['arm']==arm)
   p=Path(row['feature_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['feature_sha256']
   x=torch.load(p,map_location='cpu',weights_only=True).numpy().astype(float)
   assert x.shape==(5000,2048) and np.isfinite(x).all()
   arrays.append(x);parents.append(Path(row['sample_path']).parent)
  if group=='sit5k':
   a,b=[json.loads((p/'summary.json').read_text()) for p in parents]
   for k in ['noise_sha256','label_sha256','seed','samples','batch_size']:assert a[k]==b[k]
  else:
   for rank in range(4):
    a,b=[json.loads((p/f'sampling_audit_rank{rank}.json').read_text()) for p in parents]
    for k in ['sampling_seed','sample_count','per_rank_batch','initial_generator_sha256','final_generator_sha256','first_noise_sha256','first_label_sha256']:assert a[k]==b[k]
  delta=arrays[1]-arrays[0]
  ids=np.random.default_rng(202609436).permutation(5000);parts=[ids[:2500],ids[2500:]]
  means=[delta[i].mean(0) for i in parts]
  cosine=float(means[0]@means[1]/np.linalg.norm(means[0])/np.linalg.norm(means[1]))
  folds=[]
  for fit,test in [(0,1),(1,0)]:
   direction=means[fit]/np.linalg.norm(means[fit]);projections=delta[parts[test]]@direction
   abs_values=np.abs(projections);top=np.sort(abs_values)[-25:].sum()/abs_values.sum()
   folds.append(dict(fit_half=fit,test_half=test,mean_projection=float(projections.mean()),
                     projection_quantiles=np.quantile(projections,[0,.1,.25,.5,.75,.9,1]).tolist(),
                     positive_projection_fraction=float((projections>0).mean()),
                     largest_one_percent_share_of_absolute_projection=float(top)))
  result=dict(group=group,split_seed=202609436,split_sha256=hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest(),
              half_mean_cosine=cosine,half_mean_squared_norm=[float(m@m) for m in means],
              cross_half_mean_inner_product=float(means[0]@means[1]),folds=folds)
  results.append(result);print(json.dumps(result),flush=True)
 with (R/'pfr_mean_shift_crossfit.json').open('x') as f:json.dump(dict(complete=True,results=results,scope=__doc__),f,indent=2)
if __name__=='__main__':main()
