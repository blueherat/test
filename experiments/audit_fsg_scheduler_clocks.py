"""Execute the released FSG round-trip method with constant-noise CPU stubs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace,MethodType
import diffusers
from diffusers import DDIMScheduler,DDIMInverseScheduler
import torch


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    path=a.repo/'utils/pipeline_stable_diffusion_xl.py'
    source=path.read_text();tree=ast.parse(source)
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef)
                for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='foresight_sampling_update')
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    namespace={'torch':torch};exec(compile(module,str(path),'exec'),namespace)
    results=[]
    for interval in [0,1,5,10]:
        scheduler=DDIMScheduler(num_train_timesteps=1000,clip_sample=False)
        inverse=DDIMInverseScheduler.from_config(scheduler.config)
        scheduler.set_timesteps(40);inverse.set_timesteps(40)
        t=scheduler.timesteps[0];target=scheduler.timesteps[interval]
        x=torch.linspace(-1,1,32).reshape(2,1,4,4);eps=torch.full_like(x,.125)
        stub=SimpleNamespace(scheduler=scheduler,inv_scheduler=inverse,
            noise_pred=lambda *args:eps,apply_guidance=lambda pred,scale:pred)
        fn=MethodType(namespace['foresight_sampling_update'],stub)
        actual=fn(x.clone(),t,target,5.5,0.,None,None,1)
        def transfer(z,from_t,to_t):
            alpha=scheduler.alphas_cumprod
            clean=(z-(1-alpha[from_t]).sqrt()*eps)/alpha[from_t].sqrt()
            return alpha[to_t].sqrt()*clean+(1-alpha[to_t]).sqrt()*eps
        stride=1000//40
        expected=transfer(transfer(x,int(t),int(t)-stride),int(target)-stride,int(target))
        aligned=transfer(transfer(x,int(t),int(target)),int(target),int(t))
        assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6)
        assert torch.allclose(aligned,x,atol=2e-6,rtol=2e-6)
        if interval==0:assert torch.allclose(actual,x,atol=2e-6,rtol=2e-6)
        else:assert (actual-x).abs().max()>1e-4
        results.append({'interval':interval,'nominal_start':int(t),'nominal_target':int(target),
            'actual_forward_from':int(t),'actual_forward_to':int(t)-stride,
            'actual_inverse_from':int(target)-stride,'actual_inverse_to':int(target),
            'roundtrip_rms':float((actual-x).square().mean().sqrt()),
            'explicit_formula_max_error':float((actual-expected).abs().max()),
            'aligned_roundtrip_max_error':float((aligned-x).abs().max())})
    result={'complete':True,'diffusers_version':diffusers.__version__,
        'repo_commit':subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope':'released method with real local DDIM schedulers, constant epsilon, synthetic CPU latents; not authors experimental environment or image-quality reproduction',
        'rows':results}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('/home/zhoushunyu/data/research_repos/Foresight-Guidance'))
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
