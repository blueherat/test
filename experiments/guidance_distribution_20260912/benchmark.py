"""Same-card inference timing and native-output parity, without retraining."""
import argparse
import math
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c,ig
from . import mixture as m


def native_field(rt,z,t,left,track):
    a=c.amount(rt,left,track)
    if not a:return rt.field(z,t,'full')
    if track=='ig':s,w=rt.pair(z,t)
    else:
        s=rt.field(z,t,'full');labels=rt.labels
        rt.labels=torch.full_like(labels,c.CLASSES[rt.name])
        try:w=rt.field(z,t,'full')
        finally:rt.labels=labels
    return s+a*(s-w)


@torch.inference_mode()
def run(model):
    m.configure();rt=c.runtime(model);records=[]
    for track in ('ig','cfg'):
        m.verify(m.ROOT/model/m.stage(track)/'request.json')
        hp=m.head_path(model,track);head=m.load_head(model,track,'cuda')
        kappa=m.coefficient(model,track)
        with np.load(m.ROOT/model/(track+'_contamination_profile')/'profile.npz') as d:profile=d['gain']
        noise,_,labels=c.bank(model,m.stage(track));z=c.cuda(noise[:rt.batch]);y=c.cuda(labels[:rt.batch])
        def sample(kind):
            capture=None
            if kind!='original_native':capture=ig.Capture(rt)
            def field(x,t,left,i,j):
                if kind=='original_native':return native_field(rt,x,t,left,track)
                trace=[]
                return m.query(rt,capture,head,'native_base' if kind=='native_with_diagnostics' else 'inverse',
                               track,kappa,profile,x,t,left,i,j,trace)
            torch.cuda.synchronize();begin=time.perf_counter()
            try:
                with rt.context():latent,counts=c.integrate(rt,z,y,field)
                pixels=rt.decode(latent);torch.cuda.synchronize()
                return latent.clone(),pixels,counts,time.perf_counter()-begin
            finally:
                if capture is not None:capture.close()
        a,pa,ca,_=sample('original_native');b,pb,cb,_=sample('native_with_diagnostics')
        assert torch.equal(a,b),(a-b).abs().max()
        np.testing.assert_array_equal(pa,pb);assert ca==cb and ca['prefix']==0
        sample('inverse')
        times={'original_native':[],'inverse':[]}
        for order in (('original_native','inverse'),('inverse','original_native'),('original_native','inverse')):
            for kind in order:
                _,_,counts,seconds=sample(kind);assert counts==ca
                times[kind].append(seconds)
        med={k:float(np.median(v)) for k,v in times.items()}
        rho=torch.tensor([.01,.1,.2,.5,2/3,.8,1.,2.],dtype=torch.float64)
        actual,invalid,capped=m.gain(math.log(kappa)-rho.log(),kappa)
        bounded=rho.clamp(max=2/3);expected=bounded/(1-bounded)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=0)
        records.append(dict(model=model,track=track,batch=rt.batch,gpu=torch.cuda.get_device_name(),
            original_native_output_exact=True,original_native_pixels_exact=True,counts=ca,
            sample_and_decode_seconds=times,median_seconds=med,
            relative_median_change=med['inverse']/med['original_native']-1,
            repetitions=3,parameter_count=sum(p.numel() for p in head.parameters()),
            head_sha256=c.sha(hp),gain_algebra_passed=True,diagnostic_head_excluded_from_original_baseline=True,
            quality_workers_use_classifier_diagnostics_for_all_arms=True))
        print(model,track,'timing',med,records[-1]['relative_median_change'],flush=True)
    c.atomic(m.ROOT/model/'inference_benchmark.json',dict(complete=True,records=records,
        source_sha256=c.sha(Path(__file__)),strong_assets={str(p):c.sha(p) for p in c.asset_paths(model)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True);a=p.parse_args();run(a.model)
