import time
wall_start=time.perf_counter(); cpu_start=time.process_time()
import numpy as np, json, hashlib, math
from pathlib import Path
from scipy.special import gammaln
p=Path(__file__).parent
request=json.loads((p/'radius_request.json').read_text())
f=Path(request['source']['path'])
assert hashlib.sha256(f.read_bytes()).hexdigest()==request['source']['sha256']
a=np.load(f); e=a['clean_norm_squared_per_dimension']; r=np.sqrt(e);d=262144
assert len(e)==1000 and np.isfinite(e).all() and (e>0).all()
assert len(np.unique(a['labels']))==1000
stats=lambda x:dict(mean=float(x.mean()),sample_std=float(x.std(ddof=1)),cv=float(x.std(ddof=1)/x.mean()),quantile_probs=[0,.05,.25,.5,.75,.95,1],quantiles=np.quantile(x,[0,.05,.25,.5,.75,.95,1]).tolist())
# E[sqrt(chi2_d/d)] computed via log-gamma; no Monte Carlo.
grmean=math.exp(.5*math.log(2/d)+gammaln((d+1)/2)-gammaln(d/2))
grcv=math.sqrt(1-grmean**2)/grmean
out={'protocol':request['protocol'],'request_sha256':hashlib.sha256((p/'radius_request.json').read_bytes()).hexdigest(),'source_sha256':request['source']['sha256'],'n':len(e),'dimension':d,'energy':stats(e),'radius_div_sqrt_dimension':stats(r),'gaussian_energy_cv':math.sqrt(2/d),'gaussian_radius_cv':grcv,'empirical_over_gaussian_energy_cv':float(e.std(ddof=1)/e.mean()/math.sqrt(2/d)),'empirical_over_gaussian_radius_cv':float(r.std(ddof=1)/r.mean()/grcv),'gaussian_moment_equivalent_dimension':float(2*e.mean()**2/e.var(ddof=1))}
# Exact formula witness: unit F, unit B at angle gamma. ADG assist is perpendicular to B, not F.
gamma=math.pi/3; omega=1.78; angle=min((omega-1)*gamma,math.pi/3)
F=np.array([1.,0.]);B=np.array([math.cos(gamma),math.sin(gamma)]);assist=(F-np.dot(F,B)*B)/math.sin(gamma)
A=math.cos(angle)*F+math.sin(angle)*assist
out['adg_norm_formula_witness']={'F':F.tolist(),'B':B.tolist(),'omega':omega,'gamma':gamma,'angle':angle,'F_norm':float(np.linalg.norm(F)),'ADG_norm':float(np.linalg.norm(A)),'formula_norm_squared':1+math.sin(2*angle)*math.sin(gamma),'direct_norm_squared':float(np.dot(A,A)),'assist_dot_F':float(np.dot(assist,F)),'assist_dot_B':float(np.dot(assist,B))}
assert abs(np.dot(A,A)-(1+math.sin(2*angle)*math.sin(gamma)))<1e-14
out['cost']={'wall_seconds':time.perf_counter()-wall_start,'cpu_seconds':time.process_time()-cpu_start,'scope':'From import time through calculation, before serialization and process exit; no GPU/model/noise/FID.'}
(p/'radius_and_geometry_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
