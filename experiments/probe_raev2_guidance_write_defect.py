"""Measure numerical roundtrip defect relative to IG/CFG write signals."""
import hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch
from experiments import sample_raev2_pfr_retiming as native

@torch.inference_mode()
def main():
    out=Path('/home/zhoushunyu/data/eqvae/experiments/guidance_write_defect_20260908');out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=native.load_config(native.DEFAULT_CONFIG)
    model=native.instantiate_from_config(cfg.stage_2).cuda().eval().requires_grad_(False)
    state=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(state['ema'],strict=True);del state
    sources={str(p):native.file_sha256(p) for p in [Path(__file__),Path(native.__file__),native.DEFAULT_CONFIG,Path(inspect.getfile(type(model)))]}
    gen=torch.Generator(device='cuda').manual_seed(202609442)
    bank=torch.randn(64,*cfg.misc.latent_size,generator=gen,device='cuda')
    labels=torch.arange(64,device='cuda')*999//63;calls=0;rows=[];started=time.perf_counter()
    def field(z,t,y):
        nonlocal calls
        ts=torch.full((len(z),),t,device='cuda');f,b=model(z,ts,context=y,attn_mask=None);calls+=1
        return tuple(native.clean_to_velocity(p,z,ts,denominator_floor=.05) for p in [f,b])
    for start in range(0,64,4):
        z=bank[start:start+4];y=labels[start:start+4];yn=torch.full_like(y,1000)
        s,w=field(z,1.,y);u,_=field(z,1.,yn)
        for name,reference,guided,ry in [('ig',s,w+1.78*(s-w),y),('cfg',u,2*s-u,yn)]:
            qg=z-.125*guided;qr=z-.125*reference
            sg,_=field(qg,.875,ry);sr,_=field(qr,.875,ry)
            tg=qg+.125*sg;tr=qr+.125*sr
            raw=tg-z;nuisance=tr-z;signal=tg-tr
            # Identical reference/guided maps cancel exactly, regardless of inverse error.
            assert torch.equal(z+(tr-tr),z)
            v=torch.stack([x.flatten(1).double() for x in [raw,nuisance,signal,-.125*(guided-reference)]],dim=1)
            gram=v@v.transpose(1,2)/v.shape[-1]
            np.savez_compressed(out/f'{name}_{start:03d}.npz',vectors=v.float().cpu().numpy())
            for j in range(4):rows.append(dict(kind=name,id=start+j,gram=gram[j].cpu().tolist()))
        print(json.dumps(dict(done=start+4,seconds=time.perf_counter()-started)),flush=True)
    assert calls==96
    result=dict(complete=True,rows=rows,sources=sources,full_batch_calls=calls,seconds=time.perf_counter()-started,
                H=.125,seed=202609442,precision='FP32 noTF32',zero_signal_identity='bitwise all64 samples in each kind',
                noise_sha256=hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
