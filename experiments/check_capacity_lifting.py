"""Time orientation, strength, zero-guidance tail and constant-field checks."""
import json
from pathlib import Path
import torch
from experiments.raev2_capacity_lifting import advance_block,blocks,lift_strength

grid=torch.linspace(1.,.6,5,dtype=torch.float64)
z=torch.tensor([[1.,-2.]],dtype=torch.float64)
S=torch.tensor([[.7,-.2]],dtype=torch.float64);W=torch.tensor([[-.3,.5]],dtype=torch.float64)
def field(z,t,kind):return (S if kind=='full' else W).expand_as(z)
errors={}
for a in [0.,.78,1.25]:
 actual=advance_block(z,grid,a,field)
 expected=z+(grid[-1]-grid[0])*(S+a*(S-W))
 errors[str(a)]=float((actual-expected).abs().max())
 assert errors[str(a)]<1.e-12
u=torch.linspace(1.,0.,101);grid2=8*u/(1+7*u)
counts={}
for schedule in ['constant','fade']:
 seg=list(blocks(grid2,.78,schedule))
 assert [j for b,e,a in seg for j in range(b,e)]==list(range(100))
 n=sum(e-b for b,e,a in seg if a)
 counts[schedule]=dict(full=100+2*n,prefix=2*n)
 for b,e,a in seg:
  if float(grid2[b])<=.2 and schedule=='fade':assert a==0.
 assert lift_strength(.2,.78,'fade')==0.
result=dict(passed=True,constant_field_IG_identity_errors=errors,expected_calls=counts,
            note='Algebra and indexing only; no image-quality claim.')
p=Path('experiments/results/terminal_defect_20260908/capacity_lifting');p.mkdir(exist_ok=True)
(p/'math_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
