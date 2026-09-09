"""Paired frozen-head scalar/PCA/random gate pilot; no Bayes training targets."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
from terminal_defect_spiral import load_setting, core, sha


class Gate(nn.Module):
    def __init__(self,dim,hidden=128,outputs=2):
        super().__init__()
        self.trunk=core.FeatureTrunk(dim,hidden,2,32)
        self.head=nn.Linear(hidden,outputs)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,z,t):
        logits=self.head(self.trunk(z,t))
        safe=t.clamp(.001,.999)
        return logits,torch.log1p(-safe)-torch.log(safe)


def mix(model,kind,basis,z,t,vx,ve):
    logits,prior=model(z,t)
    if kind=='scalar':
        g=(logits.mean(1,keepdim=True)+prior[:,None]).sigmoid()
        return ve+g*(vx-ve)
    if kind=='coordinate':
        g=(logits+prior[:,None]).sigmoid()
        return ve+g*(vx-ve)
    g=(logits+prior[:,None]).sigmoid()
    difference=vx-ve
    parallel=(difference@basis)@basis.T
    return ve+g[:,:1]*parallel+g[:,1:]*(difference-parallel)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg,dist,suite=load_setting(a.checkpoint_folder,torch.device(a.device))
    gen=torch.Generator(device=a.device).manual_seed(cfg['seed']+91823)
    with torch.no_grad():
        clean,_,_=dist.sample(1024,generator=gen)
        _,_,vh=torch.linalg.svd(clean-clean.mean(0),full_matrices=False)
        pca=vh[:2].T.contiguous()
        random=torch.linalg.qr(torch.randn(cfg['ambient_dim'],2,generator=gen,device=a.device))[0]
    torch.manual_seed(cfg['seed']+91824)
    base=Gate(cfg['ambient_dim']).to(a.device)
    models={k:copy.deepcopy(base) for k in a.kinds if k!='coordinate'}
    if 'coordinate' in a.kinds:
        torch.manual_seed(cfg['seed']+91824)
        models['coordinate']=Gate(cfg['ambient_dim'],hidden=76,outputs=cfg['ambient_dim']).to(a.device)
    bases={'scalar':pca,'pca':pca,'random':random,'coordinate':pca}
    opts={k:torch.optim.Adam(m.parameters(),lr=2e-4) for k,m in models.items()}
    core.save_json(a.output/'manifest.json',{'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'checkpoint_sha256':sha(a.checkpoint_folder/'checkpoint.pt'),'seed':cfg['seed'],
        'parameters_per_gate':sum(p.numel() for p in base.parameters()),'torch':torch.__version__,
        'actual_parameters':{k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
        'training':'paired clean-noise velocity targets only; no Bayes/subspace oracle',
        'geometry_residual':'off-subspace RMS' if cfg['curvature']==0 else 'embedding consistency RMS, not closest-manifold distance'})
    started=time.perf_counter()
    losses=[]
    for step in range(1,a.updates+1):
        with torch.no_grad():
            clean,_,_=dist.sample(1024,generator=gen)
            noise=torch.randn(clean.shape,generator=gen,device=a.device)
            t=torch.rand(1024,generator=gen,device=a.device).clamp(.001,.999)
            z=(1-t[:,None])*noise+t[:,None]*clean
            out=suite.models['D0_xeps'](z,t)
            vx,ve=core.endpoint_velocities(state=z,time_value=t,clean_prediction=out['x'],
                epsilon_prediction=out['eps'],denominator_floor=cfg['denominator_floor'])
            target=clean-noise
        row={'step':step}
        for kind,m in models.items():
            opts[kind].zero_grad(set_to_none=True)
            loss=(mix(m,kind,bases[kind],z,t,vx,ve)-target).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError(kind)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opts[kind].step()
            row[kind]=float(loss.detach())
        if step%500==0:
            row['seconds']=time.perf_counter()-started
            losses.append(row)
            print(json.dumps(row),flush=True)
    torch.save({'models':{k:m.state_dict() for k,m in models.items()},'pca':pca,'random':random},a.output/'checkpoint.pt')
    core.save_json(a.output/'training.json',losses)
    evaluation=torch.Generator(device=a.device).manual_seed(a.bank_seed)
    noise=torch.randn(4096,cfg['ambient_dim'],generator=evaluation,device=a.device)
    ref,ref_u,_=dist.sample(4096,generator=evaluation)
    original=core.condition_field
    def dispatch(condition,**kw):
        if condition not in models: return original(condition,**kw)
        z,t=kw['state'],kw['time_value']
        out=suite.models['D0_xeps'](z,t)
        vx,ve=core.endpoint_velocities(state=z,time_value=t,clean_prediction=out['x'],
            epsilon_prediction=out['eps'],denominator_floor=cfg['denominator_floor'])
        v=mix(models[condition],condition,bases[condition],z,t,vx,ve)
        return core._endpoint_override(v,velocity_x=vx,velocity_epsilon=ve,time_value=t,
            denominator_floor=cfg['denominator_floor']),None
    core.condition_field=dispatch
    rows=[]
    try:
        with torch.no_grad():
            conditions=['D4_gate_on_D0']+list(models)
            if cfg['curvature']==0:
                conditions.insert(0,'D3_oracle_bayes_gate')
            for kind in conditions:
                z,_=core.sample_heun(kind,suite=suite,distribution=dist,initial_noise=noise,
                    steps=200,denominator_floor=cfg['denominator_floor'],snapshot_times=[])
                u=dist.decode_intrinsic(z)
                row={'condition':kind,'seed':cfg['seed'],
                    'ambient_swd':core.fixed_swd(z.cpu().numpy(),ref.cpu().numpy(),directions=core.fixed_directions(cfg['ambient_dim'],256,73003)),
                    'intrinsic_swd':core.fixed_swd(u.cpu().numpy(),ref_u.cpu().numpy(),directions=core.fixed_directions(2,256,73002)),
                    'off_subspace_rms':float(dist.off_subspace_rms(z).mean())}
                rows.append(row)
                np.savez(a.output/(kind+'.npz'),endpoint=z.cpu().numpy())
                print(json.dumps(row),flush=True)
    finally:
        core.condition_field=original
    core.save_json(a.output/'summary.json',{'conditions':rows,'complete':True,'seconds':time.perf_counter()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--bank-seed',type=int,required=True)
    p.add_argument('--updates',type=int,default=5000)
    p.add_argument('--kinds',nargs='+',choices=['scalar','pca','random','coordinate'],default=['scalar','pca','random'])
    run(p.parse_args())
