"""Four cards cooperate per new lifting arm. Existing IG/PFR are read-only."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments.sample_small_sit_best_lifting_sweep_shard import ROOT,REF,DATA,ARMS,sha,atomic

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 atomic(ROOT/'status.json',dict(phase='waiting_for_historical_anchor',time=time.time()))
 prior=ROOT.parent/'small_sit_best_config_lifting_20260909/status.json'
 while True:
  phase=json.loads(prior.read_text())['phase'] if prior.exists() else 'starting'
  if phase=='complete':break
  if phase=='failed':raise RuntimeError('historical anchor failed; stop sweep')
  time.sleep(10)
 ROOT.mkdir(parents=True,exist_ok=True)
 for arm in ARMS:
  out=ROOT/arm;out.mkdir(exist_ok=True)
  if (out/'result.json').exists():continue
  atomic(ROOT/'status.json',dict(phase='sampling',arm=arm,time=time.time()))
  workers=[];logs=[]
  for rank in range(4):
   log=(out/f'rank{rank}.log').open('a');logs.append(log)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   workers.append(subprocess.Popen([sys.executable,'-m','experiments.sample_small_sit_best_lifting_sweep_shard','--rank',str(rank),'--arm',arm],env=env,stdout=log,stderr=subprocess.STDOUT))
  atomic(ROOT/'workers.json',dict(arm=arm,pids=[p.pid for p in workers]))
  codes=[p.wait() for p in workers]
  for log in logs:log.close()
  assert codes==[0]*4,codes
  records=[];ref=json.loads((REF/'sampling_manifest.json').read_text())
  for rank in range(4):
   rd=out/f'rank{rank}';summary=json.loads((rd/'summary.json').read_text());assert summary['complete']
   for k in ['noise_sha256','label_sha256']:assert summary[k]==ref[k]
   assert summary['request_sha256']==sha(rd/'request.json')
   for rec in summary['files']:records.append((rec['start'],rd/rec['file'],rec['sha256'],summary['request_sha256']))
  records.sort();assert [r[0] for r in records]==list(range(0,1000,8))
  pixels=[];labels=[];seconds=0.;full=prefix=0
  for start,p,h,rh in records:
   assert sha(p)==h
   with np.load(p) as z:
    assert str(z['request_sha256'])==rh
    pixels.append(z['arr_0']);labels.append(z['labels']);seconds+=float(z['seconds']);full+=int(z['full_calls']);prefix+=int(z['prefix_calls'])
  labels=np.concatenate(labels);assert hashlib.sha256(labels.tobytes()).hexdigest()==ref['label_sha256']
  np.testing.assert_array_equal(labels,np.load(REF/'labels_n1000.npy'))
  np.savez(out/'samples_n1000.npz',arr_0=np.concatenate(pixels));np.save(out/'labels.npy',labels)
  atomic(ROOT/'status.json',dict(phase='evaluating',arm=arm,time=time.time()))
  cmd=['/data/shared/envs/adm-fid/bin/python','experiments/compute_adm_fid.py','--reference',str(DATA/'adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'),'--samples',str(out/'samples_n1000.npz'),'--output',str(out/'fid.json'),'--activations-output',str(out/'activations.npz'),'--batch-size','8','--gpu-memory-fraction','.30']
  with (out/'fid.log').open('w') as log:subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TF_CPP_MIN_LOG_LEVEL='3'),stdout=log,stderr=subprocess.STDOUT,check=True)
  result=dict(complete=True,arm=arm,samples=1000,sample_sha256=sha(out/'samples_n1000.npz'),sum_batch_gpu_seconds=seconds,full_calls_per_image=full/125,prefix_calls_per_image=prefix/125,metrics=json.loads((out/'fid.json').read_text()),noise_sha256=ref['noise_sha256'],label_sha256=ref['label_sha256'])
  atomic(out/'result.json',result);print(json.dumps(result),flush=True)
 atomic(ROOT/'status.json',dict(phase='complete',time=time.time(),arms=list(ARMS)))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  atomic(ROOT/'status.json',dict(phase='failed',error=repr(e),time=time.time()));raise
