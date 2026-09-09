"""Evaluate fixed PFR directions against existing complete-suffix adjoints."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import audit_raev2_endpoint_adjoint_response as cache
from experiments import sample_raev2_pfr_retiming as native

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0)
    bank=cache.RESTART/'endpoint_adjoint_response_v1'
    records=[]
    for label in cache.IDS:
        folder=bank/'collect'/f'id{label:04d}'
        meta=json.loads((folder/'summary.json').read_text())
        assert meta['complete'] and meta['global_id']==label
        cache.verify(meta['states']);cache.verify(meta['adjoints'])
        records.append((label,folder,meta))
    assert native.file_sha256(native.DEFAULT_CHECKPOINT)==cache.EXPECTED['stage2']
    config=native.load_config(native.DEFAULT_CONFIG)
    model=native.instantiate_from_config(config.stage_2).cuda().eval().requires_grad_(False)
    ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(ck['ema'],strict=True);del ck
    grid=cache.time_grid();rows=[];full_calls=0;prefix_calls=0
    started=time.perf_counter()
    for label,folder,meta in records:
        states=np.load(folder/'states.npy',mmap_mode='r')
        adjoints=np.load(folder/'post_state_adjoint.npy',mmap_mode='r')
        labels=torch.tensor([label],device='cuda')
        for k,(t,s) in enumerate(zip(grid[:-1],grid[1:])):
            z=torch.from_numpy(np.array(states[k:k+1])).cuda()
            ts=torch.tensor([t],device='cuda')
            full,base=model(z,ts,context=labels,attn_mask=None);full_calls+=1
            clean=cache.official_heads(full,base,ts)
            following=cache.euler_from_clean(z,clean,t,s)
            # Revalidate the dynamics of the cached trajectory before using its derivative.
            assert cache.tensor_hash(following)==meta['state_sha256_by_step'][k+1]
            if t<=.5:continue
            future=max(.5,t-1/32)
            tf=torch.tensor([future],device='cuda')
            bf=native.evaluate_base_head_only(model,z,tf,context=labels,attn_mask=None);prefix_calls+=1
            w=native.clean_to_velocity(base,z,ts,denominator_floor=.05)
            wf=native.clean_to_velocity(bf,z,tf,denominator_floor=.05)
            correction=1.78*(w-wf)
            move=-(t-s)*correction
            a=torch.from_numpy(np.array(adjoints[k:k+1])).cuda()
            action=float((a.double()*move.double()).sum())
            assert np.isfinite(action)
            rows.append(dict(label=label,index=k,noise_time=t,first_variation=action,
                             move_squared=float(move.double().square().sum())))
        print(json.dumps(dict(label=label,first_variation=sum(r['first_variation'] for r in rows if r['label']==label))),flush=True)
    output=dict(rows=rows,full_calls=full_calls,prefix_calls=prefix_calls,seconds=time.perf_counter()-started,
                checkpoint_sha256=cache.EXPECTED['stage2'],source_sha256=cache.sha256(__file__),
                cache_records=[dict(label=l,summary=cache.record(p/'summary.json')) for l,p,m in records],
                scope='Derivative at amplitude zero of finite-horizon h=1/32 time-only PFR; fixed class-prototype observable, not FID or finite-amplitude quality.')
    with (ROOT/'experiments/results/terminal_defect_20260908/raev2_pfr_cached_adjoint.json').open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps({k:v for k,v in output.items() if k not in ['rows','cache_records']}),flush=True)
if __name__=='__main__':main()
