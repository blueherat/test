"""Resume JiT after SiT; train, tune one-depth IG, then PFR, then lifting."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments.sample_jit_pfr_variants_shard import ROOT,BASE,TRAIN,sha,atomic,VARIANTS
SIT=ROOT.parent/'small_sit_best_lifting_sweep_20260909/status.json'
EARLY=[0.,.35,.6,.9,1.2];LATE=[0.,.15,.3,.5]
def status(phase,**kw):atomic(ROOT/'status.json',dict(phase=phase,time=time.time(),**kw))
CURRENT_ARM=None
def name(method,e,l,it=1,n=1000,seed=202609831):return CURRENT_ARM
def evaluate(method,e,l,it=1,n=1000,seed=202609831):
 arm=name(method,e,l,it,n,seed);out=ROOT/arm;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 if method=='ig' and e==0 and l==0 and n==1000 and seed==202609831:
  met=json.loads((BASE/'full_euler100/fid.json').read_text())[0]
  r=dict(arm=arm,method=method,early=e,late=l,fid=met['fid'],reused=True,source=str(BASE/'full_euler100/fid.json'));atomic(out/'result.json',r);return r
 status('sampling',arm=arm);workers=[];logs=[]
 try:
  for rank in range(4):
   f=(out/f'rank{rank}.log').open('a');logs.append(f)
   cmd=[sys.executable,'-m','experiments.sample_jit_pfr_variants_shard','--rank',str(rank),'--arm',arm,'--method',method,'--early',str(e),'--late',str(l),'--iterations',str(it),'--depth','4','--samples',str(n),'--seed',str(seed)]
   workers.append(subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT))
  atomic(ROOT/'workers.json',dict(arm=arm,pids=[p.pid for p in workers]))
  while True:
   codes=[p.poll() for p in workers]
   if any(c is not None and c!=0 for c in codes):raise RuntimeError((arm,codes))
   if all(c==0 for c in codes):break
   time.sleep(5)
 finally:
  for p in workers:
   if p.poll() is None:p.terminate()
  for p in workers:p.wait()
  for f in logs:f.close()
 images=np.empty((n,256,256,3),np.uint8);seen=set();seconds=0.;full=prefix=0;reference=json.loads((BASE/'full_euler100/input_hashes.json').read_text()) if n==1000 and seed==202609831 else None
 for rank in range(4):
  rd=out/f'rank{rank}';s=json.loads((rd/'summary.json').read_text());req=json.loads((rd/'request.json').read_text());assert s['complete'] and sha(rd/'request.json')==s['request_sha256']
  if reference is None:reference={'noise':s['noise_sha256'],'labels':s['label_sha256']}
  assert s['noise_sha256']==reference['noise'] and s['label_sha256']==reference['labels']
  for p,h in req['sources'].items():assert sha(p)==h
  for rec in s['files']:
   k=rec['start'];assert k not in seen;seen.add(k);f=rd/rec['file'];assert sha(f)==rec['sha256']
   with np.load(f) as b:
    assert str(b['request_sha256'])==s['request_sha256'];np.testing.assert_array_equal(b['labels'],np.arange(k,k+4)%1000);images[k:k+4]=b['arr_0'];seconds+=float(b['seconds']);full+=int(b['full_calls']);prefix+=int(b['prefix_calls'])
 assert seen==set(range(0,n,4));np.savez(out/'samples.npz',arr_0=images)
 status('evaluating',arm=arm)
 with (out/'evaluation.log').open('w') as f:
  subprocess.run([sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',arm+'='+str(out/'samples.npz'),'--output',str(out/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(out/'features')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=f,stderr=subprocess.STDOUT,check=True)
 met=json.loads((out/'fid.json').read_text())[0]
 r=dict(arm=arm,method=method,early=e,late=l,iterations=it,samples=n,seed=seed,noise_sha256=reference['noise'],label_sha256=reference['labels'],fid=met['fid'],metrics=met,sum_batch_gpu_seconds=seconds,full_calls_per_image=full/(n/4),prefix_calls_per_image=prefix/(n/4),sample_sha256=sha(out/'samples.npz'))
 atomic(out/'result.json',r);print(json.dumps(r),flush=True);return r

def main():
 global CURRENT_ARM
 ROOT.mkdir(parents=True,exist_ok=True)
 assert json.loads((ROOT/'preflight.json').read_text())['passed']
 rows=[]
 for arm in VARIANTS:
  CURRENT_ARM=arm;r=evaluate('pfr',.3,0.);r['variant']=VARIANTS[arm];rows.append(r)
  atomic(ROOT/'results.json',rows)
 status('complete')
if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=repr(e));raise
