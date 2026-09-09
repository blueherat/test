"""Measure an exact three-term calibration decomposition on untouched native paths."""
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import sample_raev2_pfr_retiming as native


@torch.inference_mode()
def main():
    out=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_clock_decomposition_20260908')
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    native.install_raev2_decoder_config_compat()
    config=native.load_config(native.DEFAULT_CONFIG)
    model=native.instantiate_from_config(config.stage_2).cuda().eval().requires_grad_(False)
    ck=torch.load(native.DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(ck['ema'],strict=True)
    del ck
    gc.collect()
    grid=native.shifted_time_grid(100,math.sqrt(config.misc.time_dist_shift_dim/config.misc.time_dist_shift_base),torch.device('cuda'))
    events={int(torch.nonzero(grid<=t)[0]) for t in [1.,.875,.625]}
    floor=float(config.transport.t_eps)
    lo=float(config.guidance.ig.t_min);hi=float(config.guidance.ig.t_max)
    gen=torch.Generator(device='cuda').manual_seed(202609414)
    noise_hash=hashlib.sha256();rows=[];calls=0;started=time.perf_counter()
    def rms(x):return x.float().flatten(1).square().mean(1).sqrt()
    def cosine(x,y):
        x=x.float().flatten(1);y=y.float().flatten(1)
        return (x*y).sum(1)/(x.norm(dim=1)*y.norm(dim=1)).clamp_min(1e-20)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,32,4):
            z=torch.randn(4,*config.misc.latent_size,generator=gen,device='cuda')
            noise_hash.update(z.cpu().numpy().tobytes())
            labels=torch.arange(start,start+4,device='cuda')
            def fields(z,t):
                nonlocal calls
                ts=torch.full((len(z),),t,device='cuda')
                full,base=model(z,ts,context=labels,attn_mask=None);calls+=1
                vf=native.clean_to_velocity(full,z,ts,denominator_floor=floor)
                if lo<=t<=hi:
                    vb=native.clean_to_velocity(base,z,ts,denominator_floor=floor)
                    drift=native.pfr_velocity(vf,vb,vb,guidance_scale=1.78,revision_scale=0.)
                else:drift=vf
                g=-drift
                return g,-2*vf-g
            for i in range(100):
                t=float(grid[i]);dt=t-float(grid[i+1]);g,r=fields(z,t)
                if i in events:
                    for H in [.025,.125]:
                        q=z+.025*g
                        _,rt=fields(z,t-H);_,rq=fields(q,t-H)
                        terms={'contrast':.025*(g-r),'clock':.025*(r-rt),'state':.025*(rt-rq)}
                        total=sum(terms.values());direct=.025*(g-rq)
                        assert torch.allclose(total,direct,atol=2e-6,rtol=2e-5)
                        terms['total']=total
                        metrics={'native_rms':rms(dt*g),'latent_rms':rms(z),'identity_residual_rms':rms(total-direct)}
                        for k,v in terms.items():
                            metrics[k+'_rms']=rms(v)
                            metrics[k+'_over_native']=rms(v)/rms(dt*g).clamp_min(1e-20)
                            metrics[k+'_cos_native']=cosine(v,g)
                        for x,y in [('contrast','clock'),('contrast','state'),('clock','state')]:
                            metrics[x+'_cos_'+y]=cosine(terms[x],terms[y])
                        metrics={k:v.cpu().tolist() for k,v in metrics.items()}
                        for j in range(4):rows.append(dict(sample=start+j,index=i,noise_time=t,h=.025,H=H,**{k:v[j] for k,v in metrics.items()}))
                z=z+dt*g
            assert torch.isfinite(z).all()
            print(json.dumps({'done':start+4,'seconds':time.perf_counter()-started}),flush=True)
    with (out/'per_sample.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    sources={str(p):native.file_sha256(p) for p in [Path(__file__),Path(native.__file__),ROOT/'experiments/raev2_pfr_retiming.py',ROOT/'docs/RAEV2_CLOCK_DECOMPOSITION_PROTOCOL_20260908_ZH.md']}
    (out/'complete.json').write_text(json.dumps(dict(complete=True,samples=32,seed=202609414,seconds=time.perf_counter()-started,batched_model_calls=calls,noise_sha256=noise_hash.hexdigest(),checkpoint_sha256=native.file_sha256(native.DEFAULT_CHECKPOINT),sources=sources),indent=2))


if __name__=='__main__':main()
