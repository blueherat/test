"""Oracle-only causal control: shared scalar versus independent subspace gates.

Uses known toy data subspace, not a deployable method or a novelty claim.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terminal_defect_spiral import load_setting, core, sha


def subspace_field(condition, *, suite, distribution, state, time_value, denominator_floor):
    out=suite.models['D0_xeps'](state,time_value)
    vx,ve=core.endpoint_velocities(state=state,time_value=time_value,
        clean_prediction=out['x'],epsilon_prediction=out['eps'],denominator_floor=denominator_floor)
    target=distribution.bayes_velocity(state,time_value,denominator_floor=denominator_floor)
    basis=distribution.audit_random_basis if condition=='random_split_oracle' else distribution.basis
    xp,ep,tp=vx@basis,ve@basis,target@basis
    xn,en,tn=vx-xp@basis.T,ve-ep@basis.T,target-tp@basis.T
    gp=core.analytic_scalar_gate(xp,ep,tp,clip=True)
    gn=core.analytic_scalar_gate(xn,en,tn,clip=True)
    if condition=='coordinate_oracle':
        v=torch.maximum(torch.minimum(target,torch.maximum(vx,ve)),torch.minimum(vx,ve))
    elif condition=='ball_oracle':
        center=(vx+ve)*.5
        radius=(vx-ve).norm(dim=1,keepdim=True)*.5
        displacement=target-center
        v=center+displacement*(radius/displacement.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp_max(1.)
    elif condition=='intrinsic_scalar':
        v=gp*vx+(1-gp)*ve
    elif condition=='normal_scalar':
        v=gn*vx+(1-gn)*ve
    elif condition in ('split_oracle','random_split_oracle'):
        v=(gp*xp+(1-gp)*ep)@basis.T+gn*xn+(1-gn)*en
    else:
        raise ValueError(condition)
    return core._endpoint_override(v,velocity_x=vx,velocity_epsilon=ve,
        time_value=time_value,denominator_floor=denominator_floor), None


@torch.no_grad()
def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg,dist,suite=load_setting(a.checkpoint_folder,torch.device(a.device))
    gen=torch.Generator(device=a.device).manual_seed(a.bank_seed)
    noise=torch.randn(a.samples,cfg['ambient_dim'],generator=gen,device=a.device)
    ref,ref_u,_=dist.sample(a.samples,generator=gen)
    original=core.condition_field
    rg=torch.Generator(device='cpu').manual_seed(73004)
    dist.audit_random_basis=torch.linalg.qr(torch.randn(cfg['ambient_dim'],2,generator=rg))[0].to(a.device)
    controls=['intrinsic_scalar','normal_scalar','split_oracle','random_split_oracle','coordinate_oracle','ball_oracle']
    def dispatch(condition,**kw):
        return subspace_field(condition,**kw) if condition in controls else original(condition,**kw)
    core.condition_field=dispatch
    rows=[]
    try:
        for condition in a.conditions:
            z,_=core.sample_heun(condition,suite=suite,distribution=dist,initial_noise=noise,
                steps=200,denominator_floor=cfg['denominator_floor'],snapshot_times=[])
            u=dist.decode_intrinsic(z)
            row={'condition':condition,'seed':cfg['seed'],'samples':a.samples,
                'intrinsic_swd':core.fixed_swd(u.cpu().numpy(),ref_u.cpu().numpy(),directions=core.fixed_directions(2,256,73002)),
                'ambient_swd':core.fixed_swd(z.cpu().numpy(),ref.cpu().numpy(),directions=core.fixed_directions(cfg['ambient_dim'],256,73003)),
                'off_subspace_rms':float(dist.off_subspace_rms(z).mean())}
            rows.append(row)
            np.savez(a.output/(condition+'.npz'),endpoint=z.cpu().numpy())
            print(json.dumps(row),flush=True)
    finally:
        core.condition_field=original
    core.save_json(a.output/'summary.json',{'conditions':rows,'checkpoint_sha256':sha(a.checkpoint_folder/'checkpoint.pt'),
        'bank_seed':a.bank_seed,'claim':'exploratory oracle control; known subspace, no practical method claim'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--bank-seed',type=int,required=True)
    p.add_argument('--samples',type=int,default=4096)
    p.add_argument('--conditions',nargs='+',default=['D3_oracle_bayes_gate','D4_gate_on_D0',
        'intrinsic_scalar','normal_scalar','split_oracle','random_split_oracle'])
    run(p.parse_args())
