"""Frozen head outputs for a spatial-correspondence premise check."""
import hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native
from experiments.audit_raev2_endpoint_adjoint_response import time_grid,official_heads,euler_from_clean

OUT=Path('/home/zhoushunyu/data/eqvae/experiments/ig_head_correspondence_20260908')
CACHE=OUT.parent/'raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/collect'
IDS=[0,142,285,428,570,713,856,999]
READ=[0,47,73,87]
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=native.load_config(native.DEFAULT_CONFIG)
    model=native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    ckhash=native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert ckhash=='723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(ck['ema'],strict=True);del ck
    sources=[Path(__file__),Path(native.__file__),Path(inspect.getfile(euler_from_clean)),native.DEFAULT_CONFIG,Path(inspect.getfile(type(model)))]
    request=dict(ids=IDS,steps=READ,checkpoint_sha256=ckhash,sources={str(p):native.file_sha256(p) for p in sources},
                 premise='unrestricted token permutations bound removable positional mismatch; fit-even/test-odd channels check')
    (OUT/'request.json').write_text(json.dumps(request,indent=2))
    grid=time_grid();files={};started=time.perf_counter()
    for i in IDS:
        bank=np.load(CACHE/f'id{i:04d}/states.npy',mmap_mode='r');meta=json.loads((CACHE/f'id{i:04d}/summary.json').read_text())
        for k in READ:
            assert sha(bank[k])==meta['state_sha256_by_step'][k]
            z=torch.from_numpy(np.array(bank[k],copy=True)).unsqueeze(0).cuda()
            t=torch.tensor([grid[k]],device='cuda',dtype=torch.float32)
            f,b=model(z,t,context=torch.tensor([i],device='cuda'),attn_mask=None)
            nxt=euler_from_clean(z,official_heads(f,b,t),grid[k],grid[k+1])
            assert sha(nxt[0].cpu().numpy())==meta['state_sha256_by_step'][k+1]
            p=OUT/f'id{i:04d}_step{k:03d}.npz'
            np.savez_compressed(p,full=f[0].cpu().numpy(),base=b[0].cpu().numpy())
            files[p.name]=native.file_sha256(p)
    result=dict(complete=True,calls=32,seconds=time.perf_counter()-started,files=files,request=request,selected_step_parity=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
