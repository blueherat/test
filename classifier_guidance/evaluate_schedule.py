"""Matched 5K evaluation of a frozen initial and learned native IG schedule."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915.models import Adapter,fingerprint
from .schedules import GuidanceSchedule,for_native_sit,native_reference


def main(args):
    rank,world=c.setup()
    run=args.output
    request=c.read(run/'request.json')
    for path,digest in request['sources'].items():
        if c.sha(path)!=digest:raise RuntimeError(f'Source changed: {path}')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state['objective']!='binary_gan_signed_schedule':raise ValueError('Wrong checkpoint')
    checkpoint_sha=c.sha(args.checkpoint)
    adapter=Adapter('sit_small');adapter.native.eval().requires_grad_(False)
    assert state['frozen']==dict(strong=fingerprint(adapter.model),weak=fingerprint(adapter.native))
    bank=c.original.model_root('sit_small')/'quality_inputs'
    noise=np.load(bank/'noise.npy',mmap_mode='r');labels=np.load(bank/'labels.npy')
    assert len(noise)>=args.samples and len(labels)>=args.samples
    assert args.samples==5000 and np.all(np.bincount(labels[:5000],minlength=100)==50)
    schedule=GuidanceSchedule(state['args']['steps'],state['args']['coefficient']).cuda().eval()
    example=torch.from_numpy(np.array(noise[:8])).cuda()
    example_labels=torch.from_numpy(np.array(labels[:8])).long().cuda()
    sampler=for_native_sit(adapter,schedule,example,example_labels)
    with torch.no_grad():
        initial=sampler(example,example_labels)
        reference=native_reference(adapter,schedule,example,example_labels)
        torch.testing.assert_close(initial,reference,rtol=0,atol=0)
    del initial,reference
    if rank==0:c.atomic(run/'parity.json',dict(passed=True,initial_native_endpoint_exact=True,
        noise_sha256=c.sha(bank/'noise.npy'),labels_sha256=c.sha(bank/'labels.npy')))
    for arm in args.arms:
        if arm=='learned':schedule.load_state_dict(state['schedule'])
        else:
            with torch.no_grad():
                schedule.coefficients.zero_()
                schedule.coefficients[:state['args']['steps']//2]=state['args']['coefficient']
        target=run/arm
        for start in tuple(range(0,args.samples,8))[rank::world]:
            if (run/'STOP_AFTER_CURRENT').exists():raise RuntimeError('Evaluation stopped by marker')
            z=torch.from_numpy(np.array(noise[start:start+8])).cuda()
            y=torch.from_numpy(np.array(labels[start:start+8])).long().cuda()
            torch.cuda.synchronize();begin=time.perf_counter()
            with torch.no_grad():
                result=sampler(z,y)
                if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite evaluation endpoint')
                pixels=adapter.pixels(result)
            torch.cuda.synchronize()
            path=target/'batches'/f'{start:05d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.tmp').open('wb') as f:
                np.savez(f,arr_0=pixels,latents=result.cpu().numpy(),labels=y.cpu().numpy(),start=start)
            path.with_suffix('.tmp').replace(path)
            c.atomic(path.with_suffix('.json'),dict(sha256=c.sha(path),start=start,samples=len(y),
                noise_sha256=c.original.array_sha(noise[start:start+len(y)]),
                seconds=time.perf_counter()-begin,checkpoint_sha256=checkpoint_sha,arm=arm))
            c.atomic(target/f'progress_rank{rank}.json',dict(start=start,samples=args.samples,updated_utc=c.now()))
            adapter.values.clear()
            del z,y,result,pixels
        c.barrier()
        if rank==0:
            images=[];records=[]
            for start in range(0,args.samples,8):
                path=target/'batches'/f'{start:05d}.npz';record=c.read(path.with_suffix('.json'))
                assert record['sha256']==c.sha(path) and record['checkpoint_sha256']==checkpoint_sha
                assert record['noise_sha256']==c.original.array_sha(noise[start:start+record['samples']])
                with np.load(path) as batch:
                    assert int(batch['start'])==start
                    np.testing.assert_array_equal(batch['labels'],labels[start:start+len(batch['labels'])])
                    images.append(batch['arr_0'])
                records.append(record)
            sample_path=target/'samples.npz'
            with sample_path.with_suffix('.tmp').open('wb') as f:np.savez(f,arr_0=np.concatenate(images))
            sample_path.with_suffix('.tmp').replace(sample_path)
            c.atomic(target/'summary.json',dict(complete=True,valid=True,n=args.samples,arm=arm,
                samples_path=str(sample_path),samples_sha256=c.sha(sample_path),
                checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_sha,
                coefficients=schedule.coefficients.tolist(),records=records,
                noise_sha256=c.sha(bank/'noise.npy'),labels_sha256=c.sha(bank/'labels.npy'),
                solver='64-step Heun; same coefficient in both stages',batch=8))
            del images
            torch.cuda.empty_cache()
            command=[c.PYTHON,'-u','-m','experiments.adversarial_weak_training_20260915.score',
                     '--stage',str(target),'--shared-gpu',os.environ['CUDA_VISIBLE_DEVICES']]
            with (target/'score_holder.log').open('w') as log:
                subprocess.run(command,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(arm,c.read(target/'metrics.json')['fid'],flush=True)
        c.barrier()
    if rank==0:
        c.atomic(run/'complete.json',dict(complete=True,arms=args.arms,samples=args.samples,
            metrics={arm:c.read(run/arm/'metrics.json')['fid'] for arm in args.arms},updated_utc=c.now()))
    if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model',choices=('sit_small',),default='sit_small')
    p.add_argument('--samples',type=int,choices=(5000,),default=5000)
    p.add_argument('--arms',choices=('initial','learned'),nargs='+',default=['initial','learned'])
    main(p.parse_args())
