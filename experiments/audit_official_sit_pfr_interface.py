"""Verify official ImageNet-1K SiT-XL PFR interfaces before quality work."""
import gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run_internal_guidance_sit_audit import load_model
from experiments.raev2_training_core import file_sha256

def prefix(model,z,t,labels):
    x=model.x_embedder(z)+model.pos_embed
    y,_=model.y_embedder(labels,model.training)
    c=model.t_embedder(t)+y
    for block in model.blocks[:model.encoder_depth]:x=block(x,c)
    return model.unpatchify(model.final_layer_xr(x,c))

@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    repo=ROOT/'research_repos/internal_guidance_study/Internal-Guidance/SiT'
    ck=Path('/home/zhoushunyu/data/eqvae/models/Internal-Guidance/official/SiT/SiT-XL-IG-ImageNet256-800EP.pt')
    model,meta=load_model(repo=repo,checkpoint_path=ck,model_name='SiT-XL/2',encoder_depth=8,state_key='ema',device=torch.device('cuda'))
    gc.collect()
    from samplers import euler_sampler
    gen=torch.Generator(device='cuda').manual_seed(202609427)
    noise=torch.randn(8,4,32,32,generator=gen,device='cuda')
    labels=torch.arange(8,device='cuda');grid=torch.linspace(1,0,101,dtype=torch.float64)
    outputs={};calls=0;prefix_calls=0;started=time.perf_counter()
    for start in [0,4]:
        initial=noise[start:start+4];ys=labels[start:start+4]
        official=euler_sampler(model,initial,ys,num_steps=100,heun=False,cfg_scale=1.).float();calls+=100
        for arm in ['full','ordinary','time_only']:
            z=initial.double()
            for i,(t,s) in enumerate(zip(grid[:-1],grid[1:])):
                ts=(torch.ones(4,device='cuda',dtype=torch.float64)*t).float()
                full,base,_=model(z.float(),ts,ys);calls+=1
                if i in [0,25,50,75]:
                    b=prefix(model,z.float(),ts,ys);prefix_calls+=1
                    assert torch.equal(base,b)
                drift=full if arm=='full' else base+1.35*(full-base)
                if arm=='time_only' and float(t)>.5:
                    future=max(.5,float(t)-1/32)
                    tf=torch.full((4,),future,device='cuda')
                    bf=prefix(model,z.float(),tf,ys);prefix_calls+=1
                    drift=drift+1.35*(base-bf)
                z=z+(s-t)*drift.double()
            assert torch.isfinite(z).all()
            if arm=='full':assert torch.equal(z.float(),official)
            outputs.setdefault(arm,[]).append(z.float().cpu().numpy())
    root=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_interface_20260908')
    root.mkdir(exist_ok=False)
    np.savez(root/'latents.npz',noise=noise.cpu().numpy(),labels=labels.cpu().numpy(),**{k:np.concatenate(v) for k,v in outputs.items()})
    result=dict(complete=True,official_full_latent_parity=True,prefix_parity=True,full_calls=calls,prefix_calls=prefix_calls,
                seconds=time.perf_counter()-started,metadata=meta,checkpoint_sha256=file_sha256(ck),
                sources={str(p):file_sha256(p) for p in [Path(__file__),repo/'models/sit.py',repo/'samplers.py']},
                latent_file_sha256=file_sha256(root/'latents.npz'),
                scope='8-image interface check only. Noise time decreases; FP32 model, FP64 state/native Euler, scale1.35, h1/32 above noise .5. No decoder or quality.')
    out=ROOT/'experiments/results/terminal_defect_20260908/official_sit_pfr_interface.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
