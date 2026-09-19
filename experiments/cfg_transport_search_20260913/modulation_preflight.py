"""Actual-model finite trajectories and zero-gain CFG endpoint parity."""
import json
from pathlib import Path
import numpy as np
import torch
from . import baselines as b, control_modulation as m
from experiments.guidance_pasted_20260912 import common as c

root=c.EXPS/'cfg_transport_search_20260913'
rt=c.runtime('sit_small')
with np.load(root/'baseline_1k/inputs.npz') as d:
    z=c.cuda(d['noise'][:8]);labels=c.cuda(d['labels'][:8])
base=b.sample(rt,z,labels,dict(kind='cfg',alpha=2.75))
zero=m.sample(rt,z,labels,dict(kind='ctrl_modulation',rho=1,alpha=2.75,K=0))
assert torch.equal(base['latents'],zero['latents'])
rows=[]
for rho in [0,1,-1]:
    result=m.sample(rt,z,labels,dict(kind='ctrl_modulation',rho=rho,alpha=2.75),True)
    assert result['counts']==dict(full=224,prefix=0)
    assert len(result['snapshots'])==5 and result['trace'].shape==(64,8,9)
    rows.append(dict(rho=rho,counts=result['counts'],finite=True,
                     final_rms=float(result['latents'].square().mean().sqrt())))
out=dict(actual_model=True,samples=8,zero_K_cfg_bitwise=True,arms=rows,
         module_sha256=c.sha(Path(m.__file__).resolve()))
path=c.WORK/'docs/research/cfg_transport_search_20260913/modulation_model_check.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
