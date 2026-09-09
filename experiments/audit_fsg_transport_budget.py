"""Total scalar guidance budget does not factor a finite transport map."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtri
from experiments.audit_fsg_exact_gaussian_transport import transport,inverse_unconditional

def main():
 rows=[]
 for m in [.5,2.,4.]:
  z0=ndtri((np.arange(8192)+.5)/8192);target=m+z0;tz0=transport(z0,m)
  for k in [1,2,4,8,16,64,256]:
   z=z0.copy();anchored=z0.copy()
   for j in range(k):
    z+=(transport(z,m)-z)/k
    anchored+=(tz0-z0)/k
   for name,state in [('recomputed_displacement',z),('frozen_displacement',anchored)]:
    u=inverse_unconditional(state,m)
    w2=float(np.mean((u-target)**2))
    rows.append(dict(class_mean=m,iterations=k,arm=name,total_scalar_budget=1.,w2_squared=w2,decoded_mean=float(u.mean()),decoded_std=float(u.std())))
    if name=='frozen_displacement':assert w2<1e-20
    elif k>1:assert w2>1e-6
 out=Path('experiments/results/terminal_defect_20260908/fsg_transport_budget.csv');pd.DataFrame(rows).to_csv(out,index=False,mode='x')
 print(pd.DataFrame(rows).query('class_mean==2').to_string(index=False))
if __name__=='__main__':main()
