"""Fixed initial latent write followed by label-free generation; quality gate."""
import argparse
from contextlib import nullcontext
import gc,hashlib,inspect,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['write_drop','full','native_ig'],required=True)
    p.add_argument('--samples',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=not a.pilot;torch.backends.cudnn.allow_tf32=not a.pilot
    cfg=native.load_config(native.DEFAULT_CONFIG);native.install_raev2_decoder_config_compat()
    decoder=native.instantiate_from_config(cfg.stage_1).cuda().eval().requires_grad_(False);del decoder.encoder
    model=native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(ck['ema'],strict=True);del ck;gc.collect()
    sources={}
    for i,path in enumerate([Path(__file__),Path(native.__file__),native.DEFAULT_CONFIG,Path(inspect.getfile(type(model)))]):
        sources[str(path)]=native.file_sha256(path);shutil.copy2(path,a.output/f'{i}_{path.name}')
    grid=native.shifted_time_grid(100,8.,torch.device('cuda'));floor=float(cfg.transport.t_eps)
    seed=202609442 if a.pilot else 202609413
    gen=torch.Generator(device='cuda').manual_seed(seed)
    if a.pilot:
        assert a.samples==8 and a.arm=='write_drop'
        bank=torch.randn(64,*cfg.misc.latent_size,generator=gen,device='cuda')
    images=[];written=[];noise_hash=hashlib.sha256();label_hash=hashlib.sha256();calls=0;post_null=0
    started=time.perf_counter()
    context=nullcontext() if a.pilot else torch.autocast('cuda',dtype=torch.bfloat16)
    with context:
        for start in range(0,a.samples,4):
            count=min(4,a.samples-start)
            z=bank[start:start+count].clone() if a.pilot else torch.randn(count,*cfg.misc.latent_size,generator=gen,device='cuda')
            y=torch.arange(start,start+count,device='cuda')
            if a.pilot:y=y*999//63
            yn=torch.full_like(y,1000)
            noise_hash.update(z.cpu().numpy().tobytes());label_hash.update(y.cpu().numpy().tobytes())
            def vel(state,t,condition,ig=False):
                nonlocal calls
                ts=torch.full((len(state),),float(t),device='cuda')
                f,b=model(state,ts,context=condition,attn_mask=None);calls+=1
                vf=native.clean_to_velocity(f,state,ts,denominator_floor=floor)
                if ig and float(cfg.guidance.ig.t_min)<=float(t)<=float(cfg.guidance.ig.t_max):
                    vb=native.clean_to_velocity(b,state,ts,denominator_floor=floor)
                    # Native IG arithmetic; no PFR revision.
                    return native.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)
                return vf
            if a.arm=='write_drop':
                for _ in range(2):
                    vu=vel(z,1.,yn);vc=vel(z,1.,y)
                    q=z-.125*(2*vc-vu)
                    z=q+.125*vel(q,.875,yn)
            if a.samples==8:written.append(z.cpu().numpy())
            for t,s in zip(grid[:-1],grid[1:]):
                condition=yn if a.arm=='write_drop' else y
                drift=vel(z,t,condition,ig=a.arm=='native_ig')
                post_null+=int((condition==1000).sum())
                z=z+(s-t)*drift
            assert torch.isfinite(z).all()
            images.append(decoder.decode(z).clamp(0,1).mul(255).permute(0,2,3,1).to('cpu',torch.uint8).numpy())
            if start==0 or (start+count)%100==0:print(json.dumps(dict(arm=a.arm,done=start+count,seconds=time.perf_counter()-started)),flush=True)
    pix=np.concatenate(images);np.savez(a.output/'samples.npz',arr_0=pix)
    if written:np.save(a.output/'written.npy',np.concatenate(written))
    assert calls==(106 if a.arm=='write_drop' else 100)*((a.samples+3)//4)
    assert post_null==(100*a.samples if a.arm=='write_drop' else 0)
    m=dict(complete=True,arm=a.arm,samples=a.samples,seed=seed,batch_size=4,pilot=a.pilot,
           precision='fp32 noTF32' if a.pilot else 'bf16 TF32',H=.125,K=2,gamma=2.,steps=100,
           full_calls=calls,post_write_null_sample_calls=post_null,noise_sha256=noise_hash.hexdigest(),
           label_sha256=label_hash.hexdigest(),sources=sources,seconds=time.perf_counter()-started,
           checkpoint_sha256=native.file_sha256(native.DEFAULT_CHECKPOINT),pixel_sha256=native.file_sha256(a.output/'samples.npz'))
    (a.output/'summary.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(m),flush=True)

if __name__=='__main__':main()
