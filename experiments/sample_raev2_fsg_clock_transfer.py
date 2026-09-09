"""Direct RAEv2 transfer with native-grid and model-call controls."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as native


@torch.inference_mode()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    sources={}
    for p in [Path(__file__),Path(native.__file__),ROOT/'experiments/raev2_pfr_retiming.py']:
        sources[str(p.resolve())]=native.file_sha256(p);shutil.copy2(p,a.output/p.name)
    torch.set_num_threads(4);torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    native.install_raev2_decoder_config_compat();config=native.load_config(native.DEFAULT_CONFIG)
    decoder=native.instantiate_from_config(config.stage_1).to(a.device).eval().requires_grad_(False)
    del decoder.encoder;torch.cuda.empty_cache()
    model=native.instantiate_from_config(config.stage_2).to(a.device).eval().requires_grad_(False)
    ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(ck['ema'],strict=True);step=int(ck['step']);del ck;gc.collect()
    steps=110 if a.arm=='ordinary110' else 100
    shift=math.sqrt(config.misc.time_dist_shift_dim/config.misc.time_dist_shift_base)
    grid=native.shifted_time_grid(steps,shift,torch.device(a.device))
    ids=np.arange(a.samples,dtype=np.int64)
    if a.native_parity:
        assert a.arm=='ordinary100' and a.samples==8
        native.sample_condition(model=model,decoder=decoder,
            condition=native.SamplingCondition('ordinary',1.78,0.,0.),config=config,
            time_grid=grid,global_ids=ids,per_rank_batch=4,sampling_seed=a.seed,
            precision='bf16',output_dir=a.output/'native',rank=0,world_size=1)
    events={int(torch.nonzero(grid<=t)[0]):k for t,k in [(1.,2),(.875,2),(.625,1)]}
    assert len(events)==3
    floor=float(config.transport.t_eps);lo=float(config.guidance.ig.t_min);hi=float(config.guidance.ig.t_max)
    gen=torch.Generator(device=a.device).manual_seed(a.seed)
    noise_hash=hashlib.sha256();label_hash=hashlib.sha256()
    images=[];calls=0;calibration_calls=0;started=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,a.samples,4):
            count=min(4,a.samples-start)
            z=torch.randn(count,*config.misc.latent_size,generator=gen,device=a.device,dtype=torch.float32)
            labels=torch.from_numpy(ids[start:start+count]%1000).to(a.device)
            noise_hash.update(z.cpu().contiguous().numpy().tobytes());label_hash.update(labels.cpu().numpy().tobytes())
            def field(state,t):
                nonlocal calls
                ts=torch.full((len(state),),float(t),device=a.device,dtype=torch.float32)
                full,base=model(state,ts,context=labels,attn_mask=None);calls+=1
                vf=native.clean_to_velocity(full,state,ts,denominator_floor=floor)
                if not lo<=float(t)<=hi:return vf,vf
                vb=native.clean_to_velocity(base,state,ts,denominator_floor=floor)
                drift=native.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)
                return drift,vf
            for i in range(steps):
                t=float(grid[i]);dt=t-float(grid[i+1])
                if not a.arm.startswith('ordinary') and i in events:
                    H=.025 if a.arm=='short' else .125
                    for _ in range(events[i]):
                        drift,_=field(z,t);g=-drift
                        q=z if a.arm=='time_only' else z+.025*g
                        qdrift,qfull=field(q,t-H);reference=-2*qfull+qdrift
                        z=z+.025*(g-reference) if a.arm=='time_only' else q-.025*reference
                        calibration_calls+=2
                drift,_=field(z,t);z=z-dt*drift
            if not torch.isfinite(z).all():raise FloatingPointError(a.arm)
            pixels=decoder.decode(z).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
            images.append(pixels)
            if start==0 or (start+count)%100==0:print(json.dumps({'arm':a.arm,'done':start+count,'seconds':time.perf_counter()-started}),flush=True)
    pixels=np.concatenate(images)
    expected=steps if a.arm.startswith('ordinary') else 110
    assert calls==expected*math.ceil(a.samples/4)
    assert calibration_calls==(0 if a.arm.startswith('ordinary') else 10*math.ceil(a.samples/4))
    np.savez(a.output/'samples.npz',arr_0=pixels)
    parity=None
    if a.native_parity:
        parity=np.array_equal(pixels,np.load(a.output/'native/images-rank00.npy'))
        if not parity:raise AssertionError('native pixel parity failed')
    meta=dict(complete=True,arm=a.arm,samples=a.samples,seed=a.seed,batch_size=4,
        steps=steps,events=[dict(index=i,noise_time=float(grid[i]),iterations=k) for i,k in events.items()],
        h=.025,H=.025 if a.arm=='short' else .125,full_calls=calls,calibration_calls=calibration_calls,
        noise_sha256=noise_hash.hexdigest(),label_sha256=label_hash.hexdigest(),seconds=time.perf_counter()-started,
        checkpoint_step=step,checkpoint_sha256=native.file_sha256(native.DEFAULT_CHECKPOINT),
        pixel_sha256=native.file_sha256(a.output/'samples.npz'),native_pixel_parity=parity,sources=sources)
    (a.output/'summary.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['ordinary100','ordinary110','short','asynchronous','time_only'],required=True)
    p.add_argument('--samples',type=int,default=1000);p.add_argument('--seed',type=int,default=202609413)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-parity',action='store_true');run(p.parse_args())
