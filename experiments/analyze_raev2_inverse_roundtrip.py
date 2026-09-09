"""Independently recompute saved inverse-state and roundtrip errors on CPU."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_inverse_roundtrip_20260908')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def mse(a,b=0):
 # Match the recorded GPU statistic: FP32 subtraction, then FP64 norm.
 difference=np.asarray(a,dtype=np.float32)-np.asarray(b,dtype=np.float32)
 return float(np.square(difference.astype(np.float64)).mean())
def check(a,b):assert np.isclose(a,b,rtol=1e-10,atol=1e-14),(a,b)
def main():
 cases=[];rows=[];seconds=0;source_sets=set()
 for shard in range(4):
  root=DATA/f'shard{shard}';d=json.loads((root/'summary.json').read_text());assert d['complete'] and d['shard']==shard
  assert d['counts']==dict(parity=200,inverse=6600,roundtrip=200)
  assert len(d['cases'])==2 and len(d['rows'])==200
  source_sets.add(tuple(sorted(d['sources'].items())))
  for i,(path,digest) in enumerate(d['sources'].items()):assert sha(root/f'{i}_{Path(path).name}')==digest
  for path,digest in d['inputs'].items():assert sha(Path(path))==digest
  for case in d['cases']:
   label=case['label'];folder=root/f'id{label:04d}'
   assert sha(folder/'inverse_states.npy')==case['inverse_states_sha256'] and sha(folder/'roundtrip_endpoint.npy')==case['roundtrip_endpoint_sha256']
   inverse=np.load(folder/'inverse_states.npy',mmap_mode='r');end=np.load(folder/'roundtrip_endpoint.npy')
   original_path=next(Path(p) for p in d['inputs'] if f'id{label:04d}' in p);original=np.load(original_path,mmap_mode='r')
   assert inverse.shape==original.shape==(101,1024,16,16) and end.shape==(1024,16,16)
   assert np.array_equal(inverse[100],original[100])
   check(mse(inverse[0],original[0]),case['noise_mse']);check(mse(inverse[0],original[0])/mse(original[0]),case['noise_relative_mse'])
   check(mse(end,original[100]),case['endpoint_mse']);check(mse(end,original[100])/mse(original[100]),case['endpoint_relative_mse'])
   chosen=[r for r in d['rows'] if r['label']==label];assert {r['step'] for r in chosen}==set(range(100))
   for row in chosen:
    k=row['step'];assert np.isfinite(inverse[k]).all()
    assert hashlib.sha256(inverse[k].tobytes()).hexdigest()==row['inverse_state_sha256']
    assert hashlib.sha256(inverse[k+1].tobytes()).hexdigest()==row['target_sha256']
    check(mse(inverse[k],original[k]),row['trajectory_error_mse']);check(mse(original[k]),row['reference_state_mse'])
   cases.append(case);rows.extend(chosen)
  seconds+=d['seconds']
 assert len(source_sets)==1 and sorted(x['label'] for x in cases)==[0,142,285,428,570,713,856,999]
 result=dict(complete=True,cases=cases,rows=rows,total_sampling_seconds=seconds,total_full_calls=28000,
             independent_saved_state_audit=True,scope='CPU recomputed all800 inverse-state errors and8 roundtrip endpoint errors; local neural residuals not independently reevaluated.')
 with (ROOT/'experiments/results/terminal_defect_20260908/raev2_inverse_roundtrip.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(dict(cases=cases,total_sampling_seconds=seconds),indent=2))
if __name__=='__main__':main()
