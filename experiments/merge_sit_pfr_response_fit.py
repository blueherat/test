"""Merge frozen response-fit statistics and solve one predefined ridge model."""
import json
import torch
from experiments.fit_sit_pfr_response_shard import ROOT
from experiments.raev2_training_core import file_sha256

def main():
 torch.set_num_threads(8);G=B=E=None;seen=set();records=[];tokens=0
 for rank in range(4):
  p=ROOT/f'rank{rank}';s=json.loads((p/'summary.json').read_text());q=json.loads((p/'request.json').read_text())
  assert s['complete'] and s['rank']==rank and s['request_sha256']==file_sha256(p/'request.json')
  assert s['statistics_sha256']==file_sha256(p/'statistics.pt')
  for name,digest in q['sources'].items():assert file_sha256(name)==digest
  st=torch.load(p/'statistics.pt',map_location='cpu',weights_only=False);assert st['request_sha256']==s['request_sha256']
  for start in st['done']:assert start not in seen and (start//4)%4==rank;seen.add(start)
  G=st['G'] if G is None else G+st['G'];B=st['B'] if B is None else B+st['B'];E=st['energy'] if E is None else E+st['energy'];tokens+=st['tokens'];records.append(s)
 assert seen==set(range(0,1000,4)) and tokens==1000*50*256
 assert len({x['noise_sha256'] for x in records})==1 and len({x['label_sha256'] for x in records})==1
 G=(G+G.T)/2;lam=1e-4*G.trace()/len(G);H=G+lam*torch.eye(len(G),dtype=G.dtype)
 chol=torch.linalg.cholesky(H);theta=torch.cholesky_solve(B,chol)
 residual=float((H@theta-B).norm()/B.norm());assert residual<1e-8
 risk=float((E-2*(theta*B).sum()+(theta*(G@theta)).sum())/(tokens*B.shape[1]));zero=float(E/(tokens*B.shape[1]));assert risk>=-1e-8 and risk<=zero+1e-8
 torch.save(dict(theta=theta.float(),theta64=theta,lambda_value=float(lam),fit_sources=records),ROOT/'response_head.pt')
 result=dict(complete=True,training_mse=risk,zero_mse=zero,relative_mse=risk/zero,normal_equation_residual=residual,lambda_value=float(lam),tokens=tokens,trajectories=1000,parameters=theta.numel(),head_sha256=file_sha256(ROOT/'response_head.pt'),sum_gpu_seconds=sum(x['seconds'] for x in records),note='Training fit only; student quality not yet measured.')
 (ROOT/'fit_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
