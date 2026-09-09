"""Write class contrast once into persistent midpoint memory; withdraw labels."""
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from experiments import sample_raev2_pfr_retiming as native

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/home/zhoushunyu/data/eqvae/experiments/fsg_memory_handoff_20260908')
CACHE = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/collect')
IDS = [0, 142, 285, 428, 570, 713, 856, 999]
STEPS = [0, 47, 73, 87]


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = native.load_config(native.DEFAULT_CONFIG)
    model = native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    state = torch.load(native.DEFAULT_CHECKPOINT, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(state['ema'], strict=True)
    del state
    assert model.num_enc_blocks == 28 and model.num_cond_tokens == 12
    n = model.s_embedder.num_patches
    assert n == 256 and cfg.conditioning.arch.num_c_tokens == 8
    sources = [Path(__file__), native.DEFAULT_CONFIG, Path(native.__file__), Path(inspect.getfile(type(model)))]
    source_hashes = {str(p): native.file_sha256(p) for p in sources}
    ckhash = native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert ckhash == '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    counts = dict(full=0, prefix14=0, suffix14_and_decoder=0)

    def prefix(z, t, y):
        kw = dict(context=y, attn_mask=None)
        h, tb = model._build_sequence(z, t, kw)
        mask = model._build_attn_mask(h, kw)
        assert torch.count_nonzero(mask) == 0
        for block in model.blocks[:14]:
            h = block(h, model.enc_rope, mask)
        counts['prefix14'] += 1
        return h, tb, mask

    def suffix(h, z, tb, mask):
        for block in model.blocks[14:model.num_enc_blocks]:
            h = block(h, model.enc_rope, mask)
        h = model.s_projector(F.silu(tb + h[:, :n]))
        x = model.x_embedder(z)
        for block in model.blocks[model.num_enc_blocks:]:
            x = block(x, h, model.dec_rope)
        counts['suffix14_and_decoder'] += 1
        return model.unpatchify(model.final_layer(x, h), model.x_patch_size)

    native.install_raev2_decoder_config_compat()
    decoder=native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False)
    del decoder.encoder
    gen=torch.Generator(device='cuda').manual_seed(202609442)
    labels=torch.arange(64,device='cuda')*999//63
    bank=torch.randn(64,*cfg.misc.latent_size,generator=gen,device='cuda')
    grid=native.shifted_time_grid(100,8.,torch.device('cuda'))
    results=[]
    for arm in ['memory','zero_memory']:
        pixels=[];endpoints=[];memories=[];started=time.perf_counter()
        limit=64 if arm=='memory' else 8
        before=counts.copy()
        for start in range(0,limit,4):
            z=bank[start:start+4].clone();y=labels[start:start+4];yn=torch.full_like(y,1000)
            t=torch.ones(4,device='cuda')
            hc,_,_=prefix(z,t,y);hu,_,_=prefix(z,t,yn)
            memory=hc[:,n+4:]-hu[:,n+4:]
            if arm=='zero_memory':memory=torch.zeros_like(memory)
            memories.append(memory.cpu().numpy())
            # No y is read inside the continuation loop.
            for current,following in zip(grid[:-1],grid[1:]):
                t=torch.full((4,),float(current),device='cuda')
                h,tb,mask=prefix(z,t,yn)
                h[:,n+4:]+=memory
                clean=suffix(h,z,tb,mask)
                drift=native.clean_to_velocity(clean,z,t,denominator_floor=float(cfg.transport.t_eps))
                z=z+(following-current)*drift
            endpoints.append(z.cpu().numpy())
            pixels.append(decoder.decode(z).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy())
            if start%16==0:print(json.dumps(dict(arm=arm,done=start+4,seconds=time.perf_counter()-started)),flush=True)
        p=np.concatenate(pixels);end=np.concatenate(endpoints);m=np.concatenate(memories)
        np.savez_compressed(OUT/f'{arm}.npz',pixels=p,endpoint=end,memory=m)
        count={k:counts[k]-before[k] for k in counts}
        assert count==dict(full=0,prefix14=102*(limit//4),suffix14_and_decoder=100*(limit//4))
        result=dict(arm=arm,samples=limit,pixel_sha256=sha(p),endpoint_sha256=sha(end),memory_sha256=sha(m),
                    seconds=time.perf_counter()-started,counts=count,all_continuation_labels_null=True)
        results.append(result)
        print(json.dumps(result),flush=True)
    result=dict(complete=True,arms=results,sources=source_hashes,checkpoint_sha256=ckhash,
                seed=202609442,noise_sha256=sha(bank.cpu().numpy()),labels=labels.cpu().tolist(),
                memory_shape=[8,1440],precision='FP32 no TF32',batch=4,steps=100,
                rule='write h_c-h_null at layer14 once at t1; each step add same memory to current null class tokens')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
