import hashlib, json, time
from pathlib import Path
import numpy as np
start=time.perf_counter(); cpu=time.process_time()
A0=np.diag([2.,.5]); A1=np.array([[1.,1.],[0.,1.]])
h=np.array([.5,.5]); delta=np.array([1.,0.]); B=[A1,np.eye(2)]
G=sum(w*b@b.T for w,b in zip(h,B)); multiplier=np.linalg.solve(G,delta)
u=[b.T@multiplier for b in B]
endpoint=sum(w*b@v for w,b,v in zip(h,B,u))
energy=sum(w*v@v for w,v in zip(h,u))
assert np.allclose(endpoint,delta,rtol=0,atol=1e-14)
assert np.isclose(energy,.8,rtol=0,atol=1e-14)
assert np.isclose(energy,delta@multiplier,rtol=0,atol=1e-14)
# Orthogonality to both basis directions of the full terminal-nullspace.
null_basis=[]
for v0 in np.eye(2):
 v1=-A1@v0
 terminal=sum(w*b@v for w,b,v in zip(h,B,[v0,v1]))
 inner=sum(w*a@v for w,a,v in zip(h,u,[v0,v1]))
 assert np.allclose(terminal,0,rtol=0,atol=1e-14)
 assert abs(inner)<1e-14
 null_basis.append({'v0':v0.tolist(),'v1':v1.tolist(),'inner_product':float(inner)})
# A gradient at the successor is not automatically a gradient at the current state.
# phi_next(y)=.5||y||^2: grad_y phi_next(A0 z)=A0 z whereas grad_z phi_next(A0 z)=A0.T A0 z.
z=np.array([1.,1.]); post_gradient=A0@z; current_gradient=A0.T@post_gradient
assert not np.array_equal(post_gradient,current_gradient)
summary={'complete':True,'A0':A0.tolist(),'A1':A1.tolist(),'h':h.tolist(),'delta':delta.tolist(),
 'gram':G.tolist(),'multiplier':multiplier.tolist(),'controls':[v.tolist() for v in u],
 'terminal_displacement':endpoint.tolist(),'energy':float(energy),'null_basis':null_basis,
 'dual_endpoint_contributions':[float(w*multiplier@b@v) for w,b,v in zip(h,B,u)],
 'post_state_gradient':post_gradient.tolist(),'current_state_composite_gradient':current_gradient.tolist(),
 'guarantee':'Exact minimum energy for the fixed affine discrete terminal-mean constraint only; not a FID or nonlinear-distribution guarantee.',
 'gpu_calls':0,'wall_excluding_imports':time.perf_counter()-start,'cpu_excluding_imports':time.process_time()-cpu,
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(__file__).with_name('summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
