"""Real SiT checks before the equal-budget Z screen; no decoder or FID."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from experiments.cfg_inverse_prior_20260913.bank import NativeRuntime, sha, source_hashes
from experiments.cfg_transport_search_20260913 import baselines as b
from experiments.fm_common_inverse_copy_20260913.run import atomic
from . import sampler
from .configs import configurations

ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/z_sampling_identity_20260913/model_check')


@torch.inference_mode()
def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    rt=NativeRuntime('cuda:0')
    noise=np.random.default_rng(2026091482).standard_normal((4,4,32,32),dtype=np.float32)
    z=torch.from_numpy(noise).cuda();labels=torch.tensor([0,24,58,99],device='cuda')
    rows=[];outputs={}
    for config in configurations():
        result=sampler.sample(rt,z,labels,config,snapshots=True)
        assert result['counts']==dict(full=224,prefix=0),(config,result['counts'])
        assert torch.isfinite(result['latents']).all()
        rows.append(dict(config=config,counts=result['counts'],
                         max_abs=float(result['latents'].abs().max()),
                         snapshots=list(result['snapshots'])))
        outputs[config['arm']]=result['latents'].cpu().numpy()
    for kind,alpha in [('cfg',1.25),('apg',2.),('ctrl',2.75)]:
        cfg=dict(kind=kind,alpha=alpha,steps=64,cutoff=.75,beta=-.5,lambda_ctrl=5.,K=.2)
        native=b.sample(rt,z,labels,cfg)['latents']
        wrapped=sampler.sample(rt,z,labels,cfg)['latents']
        assert torch.equal(native,wrapped),kind
    identity=[]
    for t in (0.,.3125,.625):
        for w in (0.,1.):
            before=rt.counts['full']
            actual,stages=sampler.z_event(rt,z,t,1/56,labels,
                strong_w=w,backward_w=w,kind='z_anchored_euler')
            used=rt.counts['full']-before
            # The helper's first query is the same-field velocity used by
            # the native one-step comparison; no new reference query needed.
            expected=z+(1/56)*stages['first']
            assert torch.equal(actual,expected),(t,w)
            assert torch.equal(stages['reflected'],z),(t,w)
            assert used==3,(t,w,used)
            identity.append(dict(time=t,weight=w,same_field_exact=True,full_calls=used))
    np.savez(ROOT/'outputs.npz',noise=noise,labels=labels.cpu().numpy(),**outputs)
    summary=dict(passed=True,model=rt.metadata,samples_per_config=4,configs=rows,
                 same_field=identity,native_cfg_apg_ctrl_bitwise=True,
                 seconds=time.monotonic()-start,branch_image_evaluations=rt.branch_image_evaluations,
                 source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),
                     Path(sampler.__file__).resolve(),Path(__file__).with_name('configs.py')]},
                 runtime_sources_sha256=source_hashes(),output_sha256=sha(ROOT/'outputs.npz'),
                 interpretation='Actual-model operation/finite-state/cost verification; no image-quality conclusion.')
    atomic(ROOT/'summary.json',summary)
    print(json.dumps(dict(passed=True,configurations=len(rows),same_field_cases=len(identity),
                         seconds=summary['seconds'],branch_image_evaluations=rt.branch_image_evaluations)),flush=True)


if __name__=='__main__':main()
