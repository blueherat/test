"""Paired native-IG/Full histories: separate head contrast from state contrast."""
import hashlib, inspect, json, time
from pathlib import Path
import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native
from experiments.audit_raev2_endpoint_adjoint_response import time_grid, euler_from_clean, official_heads

OUT=Path('/home/zhoushunyu/data/eqvae/experiments/ig_guidance_history_20260908')
CACHE=OUT.parent/'raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/collect'
IDS=[0,142,285,428,570,713,856,999]
READ=[0,47,73,87]
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=native.load_config(native.DEFAULT_CONFIG)
    model=native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    checkpoint_sha=native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert checkpoint_sha=='723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(ck['ema'],strict=True);del ck
    sources=[Path(__file__),Path(native.__file__),Path(inspect.getfile(euler_from_clean)),native.DEFAULT_CONFIG,Path(inspect.getfile(type(model)))]
    request=dict(ids=IDS,read_steps=READ,precision='FP32 noTF32 B1',checkpoint_sha256=checkpoint_sha,
                 sources={str(p):native.file_sha256(p) for p in sources},target='history interaction, not quality or a deployed guidance rule')
    (OUT/'request.json').write_text(json.dumps(request,indent=2))
    grid=time_grid();rows=[];files={};calls=0;started=time.perf_counter()
    for i in IDS:
        bank=np.load(CACHE/f'id{i:04d}/states.npy',mmap_mode='r')
        meta=json.loads((CACHE/f'id{i:04d}/summary.json').read_text())
        assert sha(bank[0])==meta['noise_sha256']
        z=torch.from_numpy(np.array(bank[0],copy=True)).unsqueeze(0).cuda()
        y=torch.tensor([i],device='cuda')
        for k,(t,s) in enumerate(zip(grid[:-1],grid[1:])):
            ts=torch.tensor([t],device='cuda',dtype=torch.float32)
            f0,b0=model(z,ts,context=y,attn_mask=None);calls+=1
            if k in READ:
                assert sha(bank[k])==meta['state_sha256_by_step'][k]
                zg=torch.from_numpy(np.array(bank[k],copy=True)).unsqueeze(0).cuda()
                f1,b1=model(zg,ts,context=y,attn_mask=None);calls+=1
                predicted=euler_from_clean(zg,official_heads(f1,b1,ts),t,s)
                assert sha(predicted[0].cpu().numpy())==meta['state_sha256_by_step'][k+1]
                # Same-state head contrasts and head-specific history responses.
                names=['gap_ig','gap_full','strong_history','weak_history','state_history']
                vec=torch.stack([f1-b1,f0-b0,f1-f0,b1-b0,zg-z])[:,0].cpu().numpy()
                path=OUT/f'id{i:04d}_step{k:03d}.npz'
                np.savez_compressed(path,vectors=vec)
                files[path.name]=native.file_sha256(path)
                v=vec.reshape(5,-1).astype(float);gram=v@v.T/v.shape[1]
                rows.append(dict(id=i,step=k,time=t,names=names,gram=gram.tolist()))
            z=euler_from_clean(z,f0,t,s)
        assert torch.isfinite(z).all()
        np.save(OUT/f'full_endpoint_{i:04d}.npy',z[0].cpu().numpy())
        print(json.dumps(dict(id=i,done=True,seconds=time.perf_counter()-started)),flush=True)
    result=dict(complete=True,calls=calls,seconds=time.perf_counter()-started,rows=rows,files=files,
                cache_selected_step_parity=True,request=request)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(calls=calls,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
