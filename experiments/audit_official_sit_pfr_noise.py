"""Reconstruct the actual B4 CUDA noise streams and check bank overlap."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch

def main():
 torch.set_num_threads(4);torch.cuda.set_device(0)
 banks=[];started=time.perf_counter()
 for seed,n in [(202609428,1000),(202609429,5000)]:
  gen=torch.Generator(device='cuda').manual_seed(seed)
  whole=hashlib.sha256();labels=hashlib.sha256();individual=[];first8=hashlib.sha256()
  for start in range(0,n,4):
   x=torch.randn(4,4,32,32,device='cuda',dtype=torch.float32,generator=gen).cpu().numpy()
   whole.update(x.tobytes());labels.update((np.arange(start,start+4,dtype=np.int64)%1000).tobytes())
   if start<8:first8.update(x.tobytes())
   individual.extend(hashlib.sha256(row.tobytes()).hexdigest() for row in x)
  assert len(individual)==len(set(individual))==n
  banks.append(dict(seed=seed,samples=n,noise_sha256=whole.hexdigest(),label_sha256=labels.hexdigest(),first8_noise_sha256=first8.hexdigest(),per_image_sha256=individual))
 oldroot=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_20260908')
 newroot=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_5k_20260908')
 for arm in ['ordinary100','pfr','ordinary115']:
  r=json.loads((oldroot/arm/'quality/summary.json').read_text())
  assert r['noise_sha256']==banks[0]['noise_sha256'] and r['label_sha256']==banks[0]['label_sha256']
 for arm in ['pfr','ordinary115']:
  r=json.loads((newroot/arm/'smoke/summary.json').read_text())
  assert r['noise_sha256']==banks[1]['first8_noise_sha256']
 overlap=len(set(banks[0]['per_image_sha256'])&set(banks[1]['per_image_sha256']))
 assert overlap==0
 out=dict(banks=banks,cross_bank_overlap=overlap,seconds=time.perf_counter()-started,
          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          scope='Exact B4 CUDA stream reconstruction; final 5K aggregate hashes must still be checked after completion.')
 with Path('experiments/results/terminal_defect_20260908/official_sit_pfr_noise.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps({k:v for k,v in out.items() if k!='banks'}))
 print(json.dumps([{k:v for k,v in b.items() if k!='per_image_sha256'} for b in banks]))
if __name__=='__main__':main()
