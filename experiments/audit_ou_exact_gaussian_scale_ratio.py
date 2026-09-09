"""Exact Gaussian OU defects: finite-h ratios do not identify neural errors."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.pfr_ou_semigroup_spectrum import transport_state_at_fixed_ou_coordinate,ou_degree1_retiming_velocity_defect

ROOT=Path(__file__).resolve().parents[1]


def main():
 started=time.perf_counter();z=torch.ones(1,1,dtype=torch.float64);rows=[]
 for variance in [2.,16.,64.,256.,1024.]:
  for t in [.02,.05]:
   def v(x,u):return ((u*variance-(1-u))*x)/(u*u*variance+(1-u)**2)
   values=[]
   for h in [1/32,1/16]:
    r=t+h;q=transport_state_at_fixed_ou_coordinate(z,t,r)
    value=float(ou_degree1_retiming_velocity_defect(v(z,t),v(q,r),z,t,r))
    # Independently derive relative score of the exact standardized Gaussian.
    c=np.sqrt(t*t+(1-t)**2);cr=np.sqrt(r*r+(1-r)**2)
    alpha=t/c;ar=r/cr;delta=variance-1
    current=delta*alpha**2/(1+delta*alpha**2)
    future=delta*ar**2/(1+delta*ar**2)
    closed=(1-t)/(t*c*c)*(current-alpha/ar*future)
    assert abs(value-closed)<1e-11
    values.append(value)
   rows.append(dict(data_variance=variance,data_time=t,short_defect=values[0],long_defect=values[1],
                    signed_amplitude_ratio=values[1]/values[0],direction_cosine=float(np.sign(values[0]*values[1]))))
 chosen=[r for r in rows if r['data_variance']==64]
 assert all(r['direction_cosine']==1 and r['signed_amplitude_ratio']<1.6 for r in chosen)
 result=dict(complete=True,rows=rows,seconds=time.perf_counter()-started,
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             scope='Exact Gaussian population field, no neural network. Violates an inference from finite-h amplitude ratios to neural inconsistency; not a quality test or proof about the observed networks.')
 with (ROOT/'experiments/results/terminal_defect_20260908/ou_exact_gaussian_scale_ratio.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))


if __name__=='__main__':main()
