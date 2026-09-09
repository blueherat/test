"""Wait for authorized lifting completion; create JiT baselines then train readouts."""
import json,os,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np
OUT=Path('experiments/results/terminal_defect_20260908/jit_transfer')
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/jit_transfer_20260908')
WAIT=Path('experiments/results/terminal_defect_20260908/capacity_lifting_constant/a078_append_status.json')
def status(**x):
 p=OUT/'preparation_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(time=time.time(),**x),indent=2)+'\n');q.replace(p);print(json.dumps(x),flush=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert json.loads((OUT/'implementation_check.json').read_text())['passed']
 status(phase='waiting_for_lifting')
 while True:
  w=json.loads(WAIT.read_text())
  if w['phase']=='failed':raise RuntimeError(w)
  if w['phase']=='complete':break
  time.sleep(15)
 for arm in ['full_euler100','cfg_heun50']:
  d=ROOT/arm
  if (d/'fid.json').exists():continue
  workers=[];streams=[]
  try:
   for rank in range(4):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(rank);f=(OUT/f'{arm}_rank{rank}.log').open('a');streams.append(f)
    workers.append(subprocess.Popen([sys.executable,'-m','experiments.sample_jit_baseline_shard','--rank',str(rank),'--arm',arm],env=env,stdout=f,stderr=subprocess.STDOUT))
   status(phase='baseline_sampling',arm=arm,pids=[p.pid for p in workers])
   while True:
    codes=[p.poll() for p in workers]
    if any(c is not None and c!=0 for c in codes):raise RuntimeError((arm,codes))
    if all(c==0 for c in codes):break
    time.sleep(5)
  finally:
   for p in workers:
    if p.poll() is None:p.terminate()
   for p in workers:p.wait()
   for f in streams:f.close()
  images=np.empty((1000,256,256,3),dtype=np.uint8);seen=set();hashes=[]
  for rank in range(4):
   rd=d/f'rank{rank}';s=json.loads((rd/'summary.json').read_text());req=json.loads((rd/'request.json').read_text());assert s['complete']
   assert sha(rd/'request.json')==s['request_sha256']
   for name,value in req['sources'].items():assert sha(Path(name))==value
   hashes.append((s['noise_sha256'],s['label_sha256']))
   for r in s['files']:
    start=r['start'];assert start not in seen;seen.add(start);p=rd/r['file'];assert sha(p)==r['sha256']
    with np.load(p) as b:
     assert str(b['request_sha256'])==s['request_sha256'];np.testing.assert_array_equal(b['labels'],np.arange(start,start+4));images[start:start+4]=b['arr_0']
     if arm=='cfg_heun50' and start==0:assert bool(b['official_latent_parity'])
  assert seen==set(range(0,1000,4)) and len(set(hashes))==1
  np.savez(d/'samples.npz',arr_0=images)
  (d/'input_hashes.json').write_text(json.dumps(dict(noise=hashes[0][0],labels=hashes[0][1]))+'\n')
  if arm=='cfg_heun50':assert (d/'input_hashes.json').read_text()==(ROOT/'full_euler100/input_hashes.json').read_text()
  status(phase='baseline_evaluation',arm=arm);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
  subprocess.run([sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',arm+'='+str(d/'samples.npz'),'--output',str(d/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(d/'features')],env=env,check=True)
 status(phase='readout_training')
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0,1,2,3';env['OMP_NUM_THREADS']='2'
 with (OUT/'readout_training.log').open('a') as f:
  p=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','--module','experiments.train_jit_internal_readouts'],env=env,stdout=f,stderr=subprocess.STDOUT)
  status(phase='readout_training',pid=p.pid);code=p.wait();assert code==0,code
 status(phase='readouts_ready',next='Run paired IG/PFR/OU quality experiments; preparation is not research success.')
if __name__=='__main__':
 try:main()
 except BaseException as e:status(phase='failed',error=repr(e));raise
