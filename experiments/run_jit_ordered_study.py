"""Resume JiT after SiT; train, tune one-depth IG, then PFR, then lifting."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments.sample_jit_ordered_shard import ROOT,BASE,TRAIN,sha,atomic
SIT=ROOT.parent/'small_sit_best_lifting_sweep_20260909/status.json'
EARLY=[0.,.35,.6,.9,1.2];LATE=[0.,.15,.3,.5]
def status(phase,**kw):atomic(ROOT/'status.json',dict(phase=phase,time=time.time(),**kw))
def name(method,e,l,it=1):return f'{method}_e{e:.2f}_l{l:.2f}_m{it}'
def evaluate(method,e,l,it=1):
 arm=name(method,e,l,it);out=ROOT/arm;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 if method=='ig' and e==0 and l==0:
  met=json.loads((BASE/'full_euler100/fid.json').read_text())[0]
  r=dict(arm=arm,method=method,early=e,late=l,fid=met['fid'],reused=True,source=str(BASE/'full_euler100/fid.json'));atomic(out/'result.json',r);return r
 status('sampling',arm=arm);workers=[];logs=[]
 try:
  for rank in range(4):
   f=(out/f'rank{rank}.log').open('a');logs.append(f)
   cmd=[sys.executable,'-m','experiments.sample_jit_ordered_shard','--rank',str(rank),'--arm',arm,'--method',method,'--early',str(e),'--late',str(l),'--iterations',str(it)]
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
 images=np.empty((1000,256,256,3),np.uint8);seen=set();seconds=0.;full=prefix=0;reference=json.loads((BASE/'full_euler100/input_hashes.json').read_text())
 for rank in range(4):
  rd=out/f'rank{rank}';s=json.loads((rd/'summary.json').read_text());req=json.loads((rd/'request.json').read_text());assert s['complete'] and sha(rd/'request.json')==s['request_sha256']
  assert s['noise_sha256']==reference['noise'] and s['label_sha256']==reference['labels']
  for p,h in req['sources'].items():assert sha(p)==h
  for rec in s['files']:
   k=rec['start'];assert k not in seen;seen.add(k);f=rd/rec['file'];assert sha(f)==rec['sha256']
   with np.load(f) as b:
    assert str(b['request_sha256'])==s['request_sha256'];np.testing.assert_array_equal(b['labels'],np.arange(k,k+4));images[k:k+4]=b['arr_0'];seconds+=float(b['seconds']);full+=int(b['full_calls']);prefix+=int(b['prefix_calls'])
 assert seen==set(range(0,1000,4));np.savez(out/'samples.npz',arr_0=images)
 status('evaluating',arm=arm)
 with (out/'evaluation.log').open('w') as f:
  subprocess.run([sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',arm+'='+str(out/'samples.npz'),'--output',str(out/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(out/'features')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=f,stderr=subprocess.STDOUT,check=True)
 met=json.loads((out/'fid.json').read_text())[0]
 r=dict(arm=arm,method=method,early=e,late=l,iterations=it,fid=met['fid'],metrics=met,sum_batch_gpu_seconds=seconds,full_calls_per_image=full/250,prefix_calls_per_image=prefix/250,sample_sha256=sha(out/'samples.npz'))
 atomic(out/'result.json',r);print(json.dumps(r),flush=True);return r

def main():
 ROOT.mkdir(parents=True,exist_ok=True);status('waiting_for_small_sit')
 while True:
  phase=json.loads(SIT.read_text())['phase'] if SIT.exists() else 'starting'
  if phase=='complete':break
  if phase=='failed':raise RuntimeError('SiT queue failed, do not take its GPUs')
  time.sleep(10)
 if not (TRAIN/'complete.json').exists() and (ROOT/'adopt_training.json').exists():
  pid=json.loads((ROOT/'adopt_training.json').read_text())['pid']
  status('training',pid=pid,adopted=True)
  while not (TRAIN/'complete.json').exists():
   proc=Path(f'/proc/{pid}/cmdline')
   if not proc.exists() or b'train_jit_internal_readouts' not in proc.read_bytes():
    raise RuntimeError('Adopted training exited without completion; inspect before restarting')
   time.sleep(10)
 if not (TRAIN/'complete.json').exists():
  assert (TRAIN/'last.pt').exists();status('resuming_training',checkpoint=str(TRAIN/'last.pt'))
  with (ROOT/'training_resume.log').open('a') as f:
   p=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','--module','experiments.train_jit_internal_readouts'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
   status('training',pid=p.pid);code=p.wait();assert code==0,code
 assert json.loads((TRAIN/'complete.json').read_text())['step']==50000
 results={}
 def ig(e,l):
  r=evaluate('ig',e,l);results[r['arm']]=r;return r
 first=[ig(e,0.) for e in EARLY];best=min(first,key=lambda r:r['fid'])
 second=[ig(best['early'],l) for l in LATE];best=min(second,key=lambda r:r['fid'])
 third=[ig(e,best['late']) for e in EARLY];best=min(results.values(),key=lambda r:r['fid'])
 atomic(ROOT/'selected_ig.json',dict(best=best,evaluated=list(results.values()),selection='minimum observed 1K FID in predeclared coordinate search; not independent validation'))
 e,l=best['early'],best['late']
 pfr=evaluate('pfr',e,l)
 lifts=[]  # User paused lifting; IG tuning and PFR comparison only.
 atomic(ROOT/'comparison.json',dict(ig=best,pfr=pfr,lifting=lifts,lifting_status='paused_by_user',official_cfg=json.loads((BASE/'cfg_heun50/fid.json').read_text()),note='Paired exploratory 1K; selected IG settings frozen for all method comparisons.'))
 status('complete')
if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=repr(e));raise
