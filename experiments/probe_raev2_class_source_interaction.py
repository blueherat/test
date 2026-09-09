"""Quantify class/source interaction on frozen native trajectories."""
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native
from experiments.audit_raev2_endpoint_adjoint_response import time_grid

OUT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_class_source_interaction_20260908')
CACHE = OUT.parent/'raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/collect'
HEADS = OUT.parent/'ig_head_correspondence_20260908'
IDS = [0,142,285,428,570,713,856,999]
STEPS = [0,47,73,87]


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=native.load_config(native.DEFAULT_CONFIG)
    ckhash=native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert ckhash=='723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    model=native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(ck['ema'],strict=True)
    del ck
    old=json.loads((HEADS/'results.json').read_text())
    assert old['complete'] and old['request']['checkpoint_sha256']==ckhash
    sources=[Path(__file__),native.DEFAULT_CONFIG,Path(inspect.getfile(type(model))),Path(inspect.getfile(time_grid))]
    request=dict(ids=IDS,steps=STEPS,checkpoint_sha256=ckhash,precision='FP32/noTF32/B1',
                 sources={str(p):native.file_sha256(p) for p in sources},
                 scope='Same frozen states, class c versus trained null=1000. Signal decomposition only; no density, quality or novelty claim.')
    (OUT/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    grid=time_grid()
    rows=[]
    started=time.perf_counter()
    for label in IDS:
        states=np.load(CACHE/f'id{label:04d}/states.npy',mmap_mode='r')
        metadata=json.loads((CACHE/f'id{label:04d}/summary.json').read_text())
        for step in STEPS:
            assert sha(states[step])==metadata['state_sha256_by_step'][step]
            name=f'id{label:04d}_step{step:03d}.npz'
            assert native.file_sha256(HEADS/name)==old['files'][name]
            saved=np.load(HEADS/name)
            z=torch.from_numpy(np.array(states[step],copy=True)).unsqueeze(0).cuda()
            t=torch.tensor([grid[step]],device='cuda',dtype=torch.float32)
            fc,bc=model(z,t,context=torch.tensor([label],device='cuda'),attn_mask=None)
            np.testing.assert_array_equal(fc[0].cpu().numpy(),saved['full'])
            np.testing.assert_array_equal(bc[0].cpu().numpy(),saved['base'])
            fu,bu=model(z,t,context=torch.tensor([1000],device='cuda'),attn_mask=None)
            arrays=np.stack([v[0].cpu().numpy() for v in [fc,bc,fu,bu]])
            assert np.isfinite(arrays).all()
            np.savez_compressed(OUT/name,heads=arrays)
            a,b,u,v=arrays.astype(np.float64)
            dc=a-b
            du=u-v
            interaction=(a-u)-(b-v)
            assert np.max(np.abs(dc-du-interaction))<1e-10
            vectors=np.stack([dc.ravel(),du.ravel(),interaction.ravel(),(a-u).ravel(),(b-v).ravel()])
            gram=vectors@vectors.T/vectors.shape[1]
            rows.append(dict(label=label,step=step,time=float(t[0]),state_sha256=sha(states[step]),
                             file=name,file_sha256=native.file_sha256(OUT/name),
                             names=['conditional_gap','null_gap','interaction','full_class_gap','base_class_gap'],gram=gram.tolist()))
        print(json.dumps(dict(label=label,seconds=time.perf_counter()-started)),flush=True)
    result=dict(complete=True,rows=rows,full_sample_calls=64,seconds=time.perf_counter()-started,
                all_conditional_heads_bitwise_match=True,request_sha256=native.file_sha256(OUT/'request.json'))
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    main()
