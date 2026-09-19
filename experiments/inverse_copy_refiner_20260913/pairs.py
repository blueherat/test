"""Real-image anchored inverse-copy pairs, without fresh noising.

For tau in {.75,.5}, C_tau(x)=Flow_weak(1<-tau) Flow_strong(tau<-1)(x).
The target remains the original real latent x. A paired repair learner is an
additional hypothesis; this file does not establish generation-quality gains.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.fm_common_inverse_copy_20260913.run import BASE, CACHE, CHECKPOINTS, atomic, sha

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913')


def prepare(args):
    root = args.out
    root.mkdir(parents=True, exist_ok=False)
    labels = np.load(CACHE/'train_labels.npy', mmap_mode='r')
    ids = np.load(CACHE/'train_source_indices.npy', mmap_mode='r')
    moments = np.load(CACHE/'train_moments.npy', mmap_mode='r')
    rng = np.random.default_rng(2026091453)
    indices, split = [], []
    for c in range(100):
        chosen = rng.permutation(np.flatnonzero(labels == c))[:5]
        indices.extend(chosen.tolist())
        split.extend([0,0,0,0,1])
    indices = np.asarray(indices, dtype=np.int64)
    selected = np.array(moments[indices])
    posterior_noise = rng.standard_normal((500,4,32,32), dtype=np.float32)
    clean = (selected[:,:4]+selected[:,4:]*posterior_noise)*np.float32(.18215)
    source_ids = np.array(ids[indices], dtype=np.int64)
    assert len(set(source_ids.tolist())) == 500
    np.savez(root/'inputs.npz', clean=clean, labels=np.array(labels[indices], dtype=np.int64),
        source_ids=source_ids, indices=indices, split=np.asarray(split,dtype=np.int8),
        posterior_noise=posterior_noise)
    request = dict(seed=2026091453, clean_sources=500, fit_sources=400, holdout_sources=100,
        source_split='train', source_cache=str(CACHE), no_fid_validation_loaded=True,
        times=[.75,.5], integration_grid=64, solver='Heun, FP32',
        operation='C_tau(x) = G_weak[1<-tau](E_strong[tau<-1](x))',
        source_target='original real latent, not a regenerated teacher endpoint',
        strong=str(CHECKPOINTS['strong']), weak=str(CHECKPOINTS['weak']),
        guidance=0, same_condition=True, new_noise_in_copy=False,
        posterior_note='sampled once in frozen initial real-image bank',
        branch_calls_per_source=160, refinement_sources=8, refinement_grid=128,
        input_sha256=sha(root/'inputs.npz'), script_sha256=sha(__file__),
        cache_manifest_sha256=sha(CACHE/'manifest.json'),
        scope='structured reconstruction training pairs; no evidence yet of transfer to CFG outputs')
    atomic(root/'request.json',request)
    print(json.dumps(request),flush=True)


class Fields:
    def __init__(self, device):
        import torch
        from experiments.imagenet100_sit_multiscale_models import load_sit_field_model
        from experiments.train_imagenet100_sit_flow import load_official_sit_module, DEFAULT_OFFICIAL_SIT_REPO
        module,source = load_official_sit_module(DEFAULT_OFFICIAL_SIT_REPO,verify_source=True)
        self.models,self.metadata,self.calls = {},{},0
        for key in ('strong','weak'):
            model,sem,meta=load_sit_field_model(checkpoint_path=CHECKPOINTS[key],weights='ema',
                sit_module=module,source_metadata=source,device=torch.device(device))
            assert sem.prediction_target=='velocity'
            self.models[key]=model,sem
            self.metadata[key]=meta

    def call(self,key,z,t,labels):
        from experiments.imagenet100_sit_multiscale_models import evaluate_sit_field
        self.calls+=1
        model,sem=self.models[key]
        return evaluate_sit_field(model,sem,z,z.new_full((len(z),),t),labels)


def flow(fields,z,labels,key,a,b,grid):
    import torch
    steps=round(abs(b-a)*grid)
    h=(b-a)/steps
    for k in range(steps):
        t=a+k*h
        one=fields.call(key,z,t,labels)
        two=fields.call(key,z+h*one,t+h,labels)
        z=z+.5*h*(one+two)
        if not torch.isfinite(z).all() or z.abs().max()>1e4:
            raise FloatingPointError((key,t))
    return z


def pair(fields,clean,labels,grid=64):
    import torch
    before=fields.calls
    at75=flow(fields,clean,labels,'strong',1.,.75,grid)
    at50=flow(fields,at75,labels,'strong',.75,.5,grid)
    out75=flow(fields,at75,labels,'weak',.75,1.,grid)
    out50=flow(fields,at50,labels,'weak',.5,1.,grid)
    assert fields.calls-before==int(2.5*grid)
    return torch.stack((out75,out50),dim=1)


def generate(args):
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    root=args.out
    request=json.loads((root/'request.json').read_text())
    assert request['script_sha256']==sha(__file__)
    assert request['input_sha256']==sha(root/'inputs.npz')
    if (root/'pairs.npz').exists():
        raise FileExistsError('Preserving completed pairs')
    start=time.monotonic()
    atomic(root/'status.json',dict(status='loading'))
    fields=Fields(args.device)
    atomic(root/'models.json',fields.metadata)
    data=np.load(root/'inputs.npz')
    arrays,counts=[],0
    with torch.inference_mode():
        for begin in range(0,len(data['clean']),16):
            clean=torch.from_numpy(data['clean'][begin:begin+16]).to(args.device)
            labels=torch.from_numpy(data['labels'][begin:begin+16]).to(args.device)
            before=fields.calls
            corrupted=pair(fields,clean,labels)
            counts+=(fields.calls-before)*len(clean)
            arrays.append(corrupted.cpu().numpy())
            completed=begin+len(clean)
            atomic(root/'status.json',dict(status='running',completed=completed,total=500))
            print(json.dumps(dict(completed=completed,seconds=time.monotonic()-start)),flush=True)
        corrupted=np.concatenate(arrays)
        clean=torch.from_numpy(data['clean'][:8]).to(args.device)
        labels=torch.from_numpy(data['labels'][:8]).to(args.device)
        refined=pair(fields,clean,labels,128).cpu().numpy()
    diff=((refined.astype(np.float64)-corrupted[:8])**2).mean(axis=(2,3,4))
    drift=((refined.astype(np.float64)-data['clean'][:8,None])**2).mean(axis=(2,3,4))
    atomic(root/'solver_check.json',dict(output_change_mse=diff.tolist(),refined_corruption_mse=drift.tolist(),
        check_sources=8,quality_claim=False))
    np.savez(root/'pairs.npz', clean=data['clean'], corrupted=corrupted,
        labels=data['labels'],split=data['split'],source_ids=data['source_ids'],indices=data['indices'])
    summary=dict(complete=True,clean_sources=500,corrupted_pairs=1000,
        mean_corruption_mse=((corrupted.astype(np.float64)-data['clean'][:,None])**2).mean(axis=(0,2,3,4)).tolist(),
        production_equivalent_branch_image_evaluations=counts,
        check_equivalent_branch_image_evaluations=8*320,
        seconds=time.monotonic()-start,pairs_sha256=sha(root/'pairs.npz'))
    atomic(root/'pairs_summary.json',summary)
    atomic(root/'status.json',dict(status='complete',completed=500,total=500))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','generate'])
    p.add_argument('--out',type=Path,default=ROOT)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    (prepare if args.phase=='prepare' else generate)(args)


if __name__=='__main__':
    main()
