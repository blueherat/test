"""One supervised temperature calibration; no image-quality parameter selection."""
import argparse
import copy
import os
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit
from . import common as c,ig

DATA='ig_calibrated_source'
STAGE='ig_calibrated_screen_400'
PROTOCOL=c.WORK/'docs/IG_SOURCE_CALIBRATION_AMENDMENT_20260912_ZH.md'
RAW_DATA,RAW_STAGE=ig.DATA,ig.STAGE
CONFIGS=[dict(s) for s in ig.CONFIGS[:5]]


def fit(model):
    torch.set_num_threads(4)
    root=c.ROOT/model/RAW_DATA;out=c.ROOT/model/DATA;out.mkdir(parents=True,exist_ok=True)
    state=torch.load(root/'head.pt',map_location='cpu',weights_only=False)
    head=ig.Head(state['state']['mean'],state['state']['std']);head.load_state_dict(state['state']);head.eval()
    logits=[];truth=[];ids=[];inputs={}
    with torch.no_grad():
        for path in sorted((root/'features').glob('*.npz')):
            with np.load(path) as d:
                keep=d['ids']>=ig.TRAIN_N
                if not keep.any():continue
                inputs[str(path)]=c.sha(path)
                f=d['features'][keep];q=f.shape[1]
                t=np.tile(d['times'],len(f));st=np.tile(d['substages'],len(f));pid=np.repeat(d['ids'][keep],q)
                f=f.reshape(-1,f.shape[-1]).astype(np.float32)
                extra=torch.from_numpy(np.column_stack((t,st)).astype(np.float32))
                value=head.net(torch.cat(((torch.from_numpy(f)-head.mean)/head.std,extra),1)).flatten()
                logits.extend(value.numpy().astype(float));truth.extend([int(d['source'])]*len(value));ids.extend(pid)
    l=np.asarray(logits);y=np.asarray(truth);ids=np.asarray(ids);fit_mask=ids<1100
    grad=lambda b:np.mean((expit(b*l[fit_mask])-y[fit_mask])*l[fit_mask])
    hi=1.
    while grad(hi)<0 and hi<1024:hi*=2
    beta=0. if grad(0)>=0 else brentq(grad,0,hi)
    def score(mask,b):
        log=b*l[mask];p=expit(log);target=y[mask]
        return dict(n=len(p),bce=float(np.mean(np.logaddexp(0,log)-target*log)),
            brier=float(np.mean((p-target)**2)),accuracy=float(((p>=.5)==target).mean()))
    calibrated=copy.deepcopy(state)
    for key in ('net.2.weight','net.2.bias'):calibrated['state'][key]*=beta
    calibrated['inverse_temperature']=beta
    torch.save(calibrated,out/'head.pt')
    rawreq=root/'request.json'
    request=dict(model=model,method='one inverse-temperature fitted by convex binary NLL on source labels',
        inverse_temperature=beta,fit_seed_ids=[1000,1099],audit_seed_ids=[1100,1199],
        no_image_quality_used=True,no_additional_denoiser_training=True,raw_request_sha256=c.sha(rawreq),
        raw_head_sha256=c.sha(root/'head.pt'),head_sha256=c.sha(out/'head.pt'),inputs=inputs,
        sources={str(p):c.sha(p) for p in [Path(__file__).resolve(),PROTOCOL]},
        fit_raw=score(fit_mask,1),fit_calibrated=score(fit_mask,beta),audit_raw=score(~fit_mask,1),audit_calibrated=score(~fit_mask,beta),
        previously_viewed_aggregate_validation=True,quality_configs=CONFIGS)
    c.atomic(out/'request.json',request)
    # Same exploration bank, deliberately not presented as fresh confirmation.
    target=c.ROOT/model/STAGE/'inputs';target.mkdir(parents=True,exist_ok=True)
    for p in (c.ROOT/model/RAW_STAGE/'inputs').glob('*.npy'):
        q=target/p.name
        if not q.exists():os.link(p,q)
        assert c.sha(p)==c.sha(q)
    c.atomic(c.ROOT/model/STAGE/'request.json',dict(calibration_request_sha256=c.sha(out/'request.json'),
        bank_reused_from=RAW_STAGE,configs=CONFIGS,constant_controls_reused=['native_base','native_half','native_double']))
    print(model,{k:v for k,v in request.items() if k in ('inverse_temperature','audit_raw','audit_calibrated')},flush=True)


def configure():
    ig.DATA,ig.STAGE,ig.CONFIGS=DATA,STAGE,CONFIGS


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--model',choices=c.MODELS,required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=2);p.add_argument('--parent-pid',type=int,default=0);a=p.parse_args()
    if a.fit:fit(a.model)
    else:
        configure();ig.quality_worker(a.model,a.rank,a.world,a.parent_pid)
