"""Fixed CFG map with a fitted Gaussian input prior; zero extra SiT NFEs."""
from pathlib import Path
import numpy as np
import torch
from experiments.cfg_transport_search_20260913 import baselines as b
from experiments.fm_common_inverse_copy_20260913.run import sha
from .fit import ROOT

SOURCE_FILES=[Path(__file__).resolve(),Path(__file__).with_name('fit.py'),
              Path(__file__).with_name('bank.py'),Path(__file__).with_name('run.py'),Path(b.__file__).resolve()]
SOURCE_FILES += [p for p in [Path(__file__).with_name('bank_anderson.py'),
                            ROOT/'request.json',ROOT/'bank_summary.json'] if p.exists()]
SOURCE_FILES += sorted((ROOT/'fit').glob('*.npz'))+sorted((ROOT/'fit').glob('*.json'))
_PRIORS={}


@torch.inference_mode()
def sample(rt, noise, labels, config, snapshots=False):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    local=dict(config)
    original=noise
    if local['kind']=='inverse_prior':
        assert local.get('alpha')==1.25 and local.get('steps')==64 and local.get('cutoff')==.75
        path=local['prior_path'];digest=local['prior_sha256']
        key=(path,digest,str(noise.device),str(noise.dtype))
        if key not in _PRIORS:
            assert sha(path)==digest,'Prior changed after request freeze'
            with np.load(path) as d:
                means=torch.from_numpy(d['means']).to(noise)
                std=torch.from_numpy(d['std']).to(noise)
            assert means.shape==(100,4,32,32) and std.shape==(4,32,32)
            assert torch.isfinite(means).all() and torch.isfinite(std).all() and (std>0).all()
            _PRIORS[key]=(means,std)
        means,std=_PRIORS[key]
        noise=means[labels]+std*noise
        local['kind']='cfg'
    result=b.sample(rt,noise,labels,local,snapshots=snapshots)
    if snapshots:
        result['snapshots']['original_gaussian_noise']=original.detach().clone()
        result['snapshots']['actual_initial_noise']=noise.detach().clone()
    return result
