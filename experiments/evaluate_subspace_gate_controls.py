"""Check whether learned subspace mixing adds beyond simple PCA projection."""
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_subspace_gate_pilot import Gate,mix,load_setting,core,sha


@torch.no_grad()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg,dist,suite=load_setting(a.checkpoint_folder,torch.device(a.device))
    saved=torch.load(a.gate_folder/'checkpoint.pt',map_location=a.device,weights_only=False)
    models={k:Gate(cfg['ambient_dim'],hidden=s['head.weight'].shape[1],outputs=s['head.weight'].shape[0]).to(a.device).eval()
            for k,s in saved['models'].items()}
    for k,m in models.items():m.load_state_dict(saved['models'][k])
    p=saved['pca']
    gen=torch.Generator(device=a.device).manual_seed(a.bank_seed)
    noise=torch.randn(a.samples,cfg['ambient_dim'],generator=gen,device=a.device)
    ref,ref_u,_=dist.sample(a.samples,generator=gen)
    original=core.condition_field
    def dispatch(condition,**kw):
        if condition in ('D0_x_shared','D4_gate_on_D0'):return original(condition,**kw)
        z,t=kw['state'],kw['time_value']
        out=suite.models['D0_xeps'](z,t)
        vx,ve=core.endpoint_velocities(state=z,time_value=t,clean_prediction=out['x'],
            epsilon_prediction=out['eps'],denominator_floor=cfg['denominator_floor'])
        if condition=='projected_x':
            v=((out['x']@p)@p.T-z)/(1-t[:,None]).clamp_min(cfg['denominator_floor'])
        elif condition=='projected_scalar':
            v=mix(models['scalar'],'scalar',p,z,t,vx,ve)
            v=(v@p)@p.T-(z-(z@p)@p.T)/(1-t[:,None]).clamp_min(cfg['denominator_floor'])
        elif condition in ('scalar','pca'):
            v=mix(models[condition],condition,p,z,t,vx,ve)
        else:raise ValueError(condition)
        # Same endpoint override as every previous dual-head condition.
        return core._endpoint_override(v,velocity_x=vx,velocity_epsilon=ve,time_value=t,
            denominator_floor=cfg['denominator_floor']),None
    core.condition_field=dispatch
    rows=[]
    try:
        for kind in ['D0_x_shared','D4_gate_on_D0','scalar','pca','projected_x','projected_scalar']:
            z,_=core.sample_heun(kind,suite=suite,distribution=dist,initial_noise=noise,
                steps=200,denominator_floor=cfg['denominator_floor'],snapshot_times=[])
            u=dist.decode_intrinsic(z)
            row={'condition':kind,'seed':cfg['seed'],'samples':a.samples,
                'ambient_swd':core.fixed_swd(z.cpu().numpy(),ref.cpu().numpy(),directions=core.fixed_directions(cfg['ambient_dim'],256,73003)),
                'intrinsic_swd':core.fixed_swd(u.cpu().numpy(),ref_u.cpu().numpy(),directions=core.fixed_directions(2,256,73002)),
                'off_subspace_rms':float(dist.off_subspace_rms(z).mean())}
            rows.append(row)
            np.savez(a.output/(kind+'.npz'),endpoint=z.cpu().numpy())
            print(json.dumps(row),flush=True)
    finally:core.condition_field=original
    core.save_json(a.output/'summary.json',{'conditions':rows,'bank_seed':a.bank_seed,
        'gate_checkpoint_sha256':sha(a.gate_folder/'checkpoint.pt'),'complete':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-folder',type=Path,required=True)
    p.add_argument('--gate-folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--bank-seed',type=int,required=True)
    p.add_argument('--samples',type=int,default=4096)
    run(p.parse_args())
