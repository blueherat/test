"""End-to-end withdrawal of external labels after initial latent calibration."""
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/zhoushunyu/data/eqvae/experiments/fsg_condition_handoff_20260908')
ARMS=['conditional','unconditional','calibrate_drop','calibrate_keep','unconditional_roundtrip','wrong_class_drop']

def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=native.load_config(native.DEFAULT_CONFIG)
    native.install_raev2_decoder_config_compat()
    decoder=native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False)
    del decoder.encoder
    model=native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    state=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(state['ema'],strict=True)
    del state
    grid=native.shifted_time_grid(100,8.,torch.device('cuda'))
    gen=torch.Generator(device='cuda').manual_seed(202609442)
    labels=torch.arange(64,device='cuda')*999//63
    bank=torch.randn(64,*cfg.misc.latent_size,generator=gen,device='cuda')
    source_paths=[Path(__file__),Path(native.__file__),native.DEFAULT_CONFIG,Path(inspect.getfile(type(model)))]
    sources={str(p):native.file_sha256(p) for p in source_paths}
    ckhash=native.file_sha256(native.DEFAULT_CHECKPOINT)
    assert ckhash=='723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
    results=[]
    for arm in ARMS:
        images=[];ends=[];written=[];calls=0;null_after_write=0;total_after_write=0
        started=time.perf_counter()
        for start in range(0,64,4):
            z=bank[start:start+4].clone();y=labels[start:start+4]
            yn=torch.full_like(y,1000)
            def velocity(x,t,condition):
                nonlocal calls
                ts=torch.full((len(x),),float(t),device='cuda')
                f,_=model(x,ts,context=condition,attn_mask=None)
                calls+=1
                return native.clean_to_velocity(f,x,ts,denominator_floor=float(cfg.transport.t_eps))
            if arm not in ['conditional','unconditional']:
                yc=(y+1)%1000 if arm=='wrong_class_drop' else y
                for iteration in range(2):
                    vu=velocity(z,1.,yn)
                    vc=velocity(z,1.,yc) if arm!='unconditional_roundtrip' else vu
                    guided=2*vc-vu
                    q=z-.125*guided
                    z=q+.125*velocity(q,.875,yn)
            written.append(z.cpu().numpy())
            condition=y if arm in ['conditional','calibrate_keep'] else yn
            for t,s in zip(grid[:-1],grid[1:]):
                drift=velocity(z,t,condition)
                total_after_write+=len(z)
                null_after_write+=int((condition==1000).sum())
                z=z+(s-t)*drift
            assert torch.isfinite(z).all()
            ends.append(z.cpu().numpy())
            pixels=decoder.decode(z).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
            images.append(pixels)
            if start%16==0: print(json.dumps(dict(arm=arm,done=start+4,seconds=time.perf_counter()-started)),flush=True)
        pix=np.concatenate(images);endpoint=np.concatenate(ends);latent=np.concatenate(written)
        extra=0 if arm in ['conditional','unconditional'] else (4 if arm=='unconditional_roundtrip' else 6)
        assert calls==(100+extra)*16
        assert total_after_write==6400
        assert null_after_write==(0 if arm in ['conditional','calibrate_keep'] else 6400)
        np.savez_compressed(OUT/f'{arm}.npz',pixels=pix,endpoint=endpoint,written=latent)
        info=dict(arm=arm,full_batch_calls=calls,full_sample_calls=calls*4,
                  null_after_write=null_after_write,total_after_write=total_after_write,
                  pixel_sha256=sha(pix),endpoint_sha256=sha(endpoint),written_sha256=sha(latent),
                  seconds=time.perf_counter()-started)
        results.append(info)
        (OUT/f'{arm}.json').write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps(info),flush=True)
    # The two continuation arms must start from exactly the same written states.
    assert results[2]['written_sha256']==results[3]['written_sha256']
    result=dict(complete=True,arms=results,sources=sources,checkpoint_sha256=ckhash,
                noise_sha256=sha(bank.cpu().numpy()),labels=labels.cpu().tolist(),seed=202609442,
                precision='FP32 noTF32',batch=4,H=.125,K=2,gamma=2.,steps=100,
                calibration='noise-time1 conditional-guided Euler forward, null Euler reverse; repeated twice',
                limitation='mechanism ablation, not full FSG reproduction or matched-budget quality comparison')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('complete',flush=True)

if __name__=='__main__':main()
