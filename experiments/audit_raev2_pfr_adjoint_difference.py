"""Fixed central differences of the cached PFR first variation; no gain selection."""
import json
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import audit_raev2_endpoint_adjoint_response as cache
from experiments import sample_raev2_pfr_retiming as native

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=cache.RESTART/'endpoint_adjoint_response_v1'
    request=json.loads((bank/'request.json').read_text())
    for identity in request['identities'].values():cache.verify(identity)
    with np.load(cache.verify(request['prototypes'])) as p:
        directions=torch.from_numpy(p['directions'].copy()).cuda()
        center=torch.from_numpy(p['center'].copy()).cuda()
    model,decoder,extractor,loads=cache.load_models(torch.device('cuda:0'),request)
    prediction_path=ROOT/'experiments/results/terminal_defect_20260908/raev2_pfr_cached_adjoint.json'
    prediction=json.loads(prediction_path.read_text())
    grid=cache.time_grid();counts=Counter();rows=[];started=time.perf_counter()
    for index,label in enumerate(cache.IDS):
        folder=bank/'collect'/f'id{label:04d}'
        meta=json.loads((folder/'summary.json').read_text())
        states=np.load(cache.verify(meta['states']),mmap_mode='r')
        noise=torch.from_numpy(np.array(states[:1])).cuda()
        labels=torch.tensor([label],device='cuda')
        values={}
        for rho in [0.,2**-12,-2**-12,2**-13,-2**-13]:
            z=noise.clone()
            for k,(t,s) in enumerate(zip(grid[:-1],grid[1:])):
                ts=torch.tensor([t],device='cuda')
                full,base=model(z,ts,context=labels,attn_mask=None);counts['full']+=1
                clean=cache.official_heads(full,base,ts)
                following=cache.euler_from_clean(z,clean,t,s)
                if rho==0:
                    assert cache.tensor_hash(following)==meta['state_sha256_by_step'][k+1]
                elif t>.5:
                    tf=torch.tensor([max(.5,t-1/32)],device='cuda')
                    bf=native.evaluate_base_head_only(model,z,tf,context=labels,attn_mask=None);counts['prefix']+=1
                    w=native.clean_to_velocity(base,z,ts,denominator_floor=.05)
                    wf=native.clean_to_velocity(bf,z,tf,denominator_floor=.05)
                    following=following-(t-s)*rho*1.78*(w-wf)
                z=following
            e=cache.endpoint(decoder,extractor,z,directions[index:index+1],center,native=False,counts=counts)
            values[rho]=e['psi']
            rows.append(dict(label=label,rho=rho,psi=e['psi'],latent_sha256=cache.tensor_hash(z)))
        expected=sum(r['first_variation'] for r in prediction['rows'] if r['label']==label)
        print(json.dumps(dict(label=label,adjoint=expected,baseline=values[0.],central_12=(values[2**-12]-values[-2**-12])/(2*2**-12),central_13=(values[2**-13]-values[-2**-13])/(2*2**-13))),flush=True)
    output=dict(rows=rows,counts=dict(counts),seconds=time.perf_counter()-started,loads=loads,
                source_sha256=cache.sha256(__file__),prediction=cache.record(prediction_path),
                scope='Two fixed central differences of continuous FP32 prototype observable; no native quality or selected gain.')
    out=ROOT/'experiments/results/terminal_defect_20260908/raev2_pfr_adjoint_difference.json'
    with out.open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(dict(complete=True,counts=dict(counts),seconds=output['seconds'])),flush=True)
if __name__=='__main__':main()
