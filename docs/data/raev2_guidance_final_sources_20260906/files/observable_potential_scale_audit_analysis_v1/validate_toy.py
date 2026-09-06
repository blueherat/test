"""Fixed small-dimensional independent finite-difference checks; CPU only."""
import time
START,CPU_START=time.perf_counter(),time.process_time()
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='4'
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import sqrtm
from fid_influence import fid_and_gradients,features_influence,stratified_covariance,paired_contrast

OUT=Path(__file__).resolve().parent
def record(p):
    p=Path(p)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def oracle(mu,s,mur,r):
    root=sqrtm(s@r)
    if np.iscomplexobj(root):
        raise AssertionError('fixed SPD toy unexpectedly complex')
    return float((mu-mur)@(mu-mur)+np.trace(s)+np.trace(r)-2*np.trace(root))
def weighted_fid(x,w,mur,r):
    assert abs(w.sum()-1)<1e-12 and np.all(w>0)
    mu=w@x
    center=x-mu
    s=len(x)/(len(x)-1)*(center*w[:,None]).T@center
    return oracle(mu,s,mur,r)

def main():
    if (OUT/'toy_validation.json').exists():
        raise FileExistsError('refusing to overwrite toy validation')
    rng=np.random.default_rng(202609104)
    labels=np.arange(15)%3
    x=rng.normal(size=(15,3))+np.array([[.5,.1,-.2],[-.3,.4,.2],[.1,-.2,.6]])[labels]
    arms={'official100':x+np.array([-.2,.3,.1]),
          'potential100':x@np.array([[.9,.08,0],[-.03,1.05,.02],[.04,0,.94]])+np.array([.1,.1,-.1]),
          'official105':x@np.array([[1.04,-.04,.01],[.02,.95,.08],[0,-.03,1.02]])+np.array([-.1,.2,.05])}
    mur=np.array([.2,-.1,.3]); ref=np.array([[1.4,.2,-.15],[.2,.8,.05],[-.15,.05,1.1]])
    records={name:features_influence(a,mur,ref) for name,a in arms.items()}
    a=records['potential100']; vector=np.array([.3,-.6,.4]); direction=np.array([[.4,.2,-.1],[.2,-.3,.05],[-.1,.05,.1]])
    tests=[]
    def check(name,analytic,numeric,step=None,tolerance=5e-6):
        error=abs(analytic-numeric)/(1+abs(analytic))
        if error>tolerance: raise AssertionError((name,analytic,numeric,error))
        tests.append({'name':name,'step':step,'analytic':float(analytic),'independent_numeric':float(numeric),'scaled_absolute_error':float(error)})
    check('fid vs independent nonsymmetric-product sqrtm',a['fid'],oracle(a['mean'],a['covariance'],mur,ref),tolerance=1e-12)
    for step in (1e-4,1e-5,1e-6):
        check('mean gradient',a['mean_gradient']@vector,(oracle(a['mean']+step*vector,a['covariance'],mur,ref)-oracle(a['mean']-step*vector,a['covariance'],mur,ref))/(2*step),step)
        check('symmetric covariance gradient',np.sum(a['covariance_gradient']*direction),(oracle(a['mean'],a['covariance']+step*direction,mur,ref)-oracle(a['mean'],a['covariance']-step*direction,mur,ref))/(2*step),step)
        for j in range(15):
            weights=np.full(15,1/15); perturb=-weights.copy(); perturb[j]+=1
            numeric=(weighted_fid(arms['potential100'],weights+step*perturb,mur,ref)-weighted_fid(arms['potential100'],weights-step*perturb,mur,ref))/(2*step)
            check(f'normalized contamination weight IF sample {j}',a['influence'][j],numeric,step)
        for c in range(3):
            i,j=np.flatnonzero(labels==c)[:2]
            w=np.full(15,1/15); perturb=np.zeros(15); perturb[i]=1; perturb[j]=-1
            plus={k:weighted_fid(v,w+step*perturb,mur,ref) for k,v in arms.items()}
            minus={k:weighted_fid(v,w-step*perturb,mur,ref) for k,v in arms.items()}
            for base in ('official100','official105'):
                _,difference_if,ratio_if,_=paired_contrast(a,records[base],labels)
                check(f'paired within-class difference {base} class {c}',difference_if[i]-difference_if[j],((plus['potential100']-plus[base])-(minus['potential100']-minus[base]))/(2*step),step)
                check(f'paired within-class relative ratio {base} class {c}',ratio_if[i]-ratio_if[j],((1-plus['potential100']/plus[base])-(1-minus['potential100']/minus[base]))/(2*step),step)
    for base in ('official100','official105'):
        result,difference_if,ratio_if,_=paired_contrast(a,records[base],labels)
        for quantity,values in (('fid_difference_candidate_minus_baseline',difference_if),('relative_improvement_one_minus_candidate_over_baseline',ratio_if)):
            # Independent pairwise-difference variance, not the producer centering formula.
            variance=0.
            for c in range(3):
                block=values[labels==c]; m=len(block)
                sample_var=sum((block[i]-block[j])**2 for i in range(m) for j in range(i+1,m))/(m*(m-1))
                variance+=sample_var/(3*3*m)
            check(f'paired stratified SE {base} {quantity}',result[quantity]['stratified_first_order_se'],np.sqrt(variance),tolerance=1e-12)
        stacked=np.column_stack([a['influence'],records[base]['influence']])
        joint,*_=stratified_covariance(stacked,labels)
        check(f'paired covariance identity {base}',result['fid_difference_candidate_minus_baseline']['stratified_first_order_se']**2,joint[0,0]+joint[1,1]-2*joint[0,1],tolerance=1e-12)
    same,*_=paired_contrast(a,a,labels)
    check('identical arms have zero paired SE',same['fid_difference_candidate_minus_baseline']['stratified_first_order_se'],0.,tolerance=1e-12)
    class_only=np.array([-.7,.2,.5])[labels]
    c,*_=stratified_covariance(class_only,labels)
    check('fixed class offsets do not create within-noise variance',c[0,0],0.,tolerance=1e-12)
    rejected=[]
    for name,s,r in [('singular sample',np.diag([1.,0.,1.]),ref),('indefinite sample',np.diag([1.,-.1,1.]),ref),('singular reference',a['covariance'],np.diag([1.,1.,0.]))]:
        try: fid_and_gradients(a['mean'],s,mur,r)
        except ValueError: rejected.append(name)
        else: raise AssertionError(f'{name} did not stop')
    try: features_influence(np.ones((3,3)),mur,ref)
    except ValueError: rejected.append('N<=dimension')
    else: raise AssertionError('rank-deficient features did not stop')
    output={'complete':True,'passed':True,'toy_seed':202609104,'dimension':3,'classes':3,'noise_replicates_per_class':5,
        'tests':tests,'checks':len(tests),'maximum_scaled_absolute_error':max(v['scaled_absolute_error'] for v in tests),
        'invalid_spd_cases_stopped':rejected,'real_5k_features_read':False,'gpu_calls':0,'model_calls':0,
        'source_records':[record(__file__),record(OUT/'fid_influence.py')],
        'wall_seconds':time.perf_counter()-START,'cpu_seconds':time.process_time()-CPU_START,
        'limits':'Local finite-dimensional derivative and paired-variance implementation checks only; not calibration of high-dimensional finite-N confidence coverage.'}
    (OUT/'toy_validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='tests'}))

if __name__=='__main__': main()
