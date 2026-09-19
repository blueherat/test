"""Frozen paired screens for the CFG transport search; independent stages.

Sampler modules expose sample(rt, noise, labels, config, snapshots=False).
Each stage freezes inputs/configurations and source identities before sampling.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.guidance_pasted_20260912 import common as c

ROOT = c.EXPS / 'cfg_transport_search_20260913'
HERE = Path(__file__).resolve()
FID_PYTHON = '/data/shared/envs/adm-fid/bin/python'
REFERENCE = c.EXPS.parent / 'imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'
MODULE = 'experiments.cfg_transport_search_20260913.runner'


def npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f, **values)
    temp.replace(path)


def prepare(args):
    root = ROOT / args.phase; root.mkdir(parents=True, exist_ok=True)
    configs = json.loads(Path(args.config).read_text())
    assert len({x['arm'] for x in configs}) == len(configs)
    module = importlib.import_module(args.sampler)
    rng = np.random.default_rng(args.seed)
    labels = (np.arange(args.samples) % 100).astype(np.int64)
    rng.shuffle(labels)
    noise = rng.standard_normal((args.samples, 4, 32, 32), dtype=np.float32)
    bank = root / 'inputs.npz'
    if bank.exists():
        with np.load(bank) as data:
            np.testing.assert_array_equal(noise, data['noise'])
            np.testing.assert_array_equal(labels, data['labels'])
    else:
        npz(bank, noise=noise, labels=labels)
    paths = {HERE, Path(module.__file__).resolve(), Path(c.__file__).resolve(),
             c.WORK/'experiments/lifting_scale_sweep_20260909.py',
             *c.source_paths('sit_small')}
    extra = getattr(module, 'SOURCE_FILES', [])
    paths.update(map(Path, extra))
    request = dict(phase=args.phase, samples=args.samples, seed=args.seed, batch=args.batch,
                   sampler=args.sampler, configs=configs,
                   sources={str(p):c.sha(p) for p in sorted(paths)},
                   assets={str(p):c.sha(p) for p in [*c.asset_paths('sit_small'), REFERENCE]},
                   inputs_sha256=c.sha(bank),
                   noise_sha256=c.array_sha(noise), labels_sha256=c.array_sha(labels),
                   reference=str(REFERENCE), model='SiT-S/2 ImageNet100 EMA',
                   interpretation='Paired search stage; choose settings here, confirm on a new independent bank.',
                   cost='Single-branch model evaluations per image; sampling/decode batch GPU seconds, evaluation separate.')
    target = root/'request.json'
    if target.exists():
        assert c.read(target) == request, 'Frozen stage changed. Use a new phase.'
    else:
        c.atomic(target, request)
    print(json.dumps(dict(prepared=True, phase=args.phase, arms=len(configs),
                          samples=args.samples, root=str(root))), flush=True)


def verify(root):
    request = c.read(root/'request.json')
    for group in ('sources', 'assets'):
        for path, digest in request[group].items():
            assert c.sha(path) == digest, ('Source/asset changed', path)
    assert c.sha(root/'inputs.npz') == request['inputs_sha256']
    return request


def gallery(images, path, labels=None):
    images = images[:32]; side = 128; cols = 8
    canvas = Image.new('RGB', (cols*side, ((len(images)+cols-1)//cols)*(side+18)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, pixel in enumerate(images):
        x, y = (i%cols)*side, (i//cols)*(side+18)
        canvas.paste(Image.fromarray(pixel).resize((side,side)), (x,y))
        if labels is not None: draw.text((x+3,y+side),str(labels[i]),fill='black')
    canvas.save(path)


def collect(root, config, request, noise, labels):
    out = root/config['arm']; images=[]; latents=[]; seconds=[]; full=[]; prefix=[]; records=[]
    for start in range(0, request['samples'], request['batch']):
        path = out/f'batch{start:06d}.npz'
        with np.load(path) as data:
            stop = min(start+request['batch'],request['samples'])
            assert int(data['start']) == start
            assert str(data['request_sha256']) == c.sha(root/'request.json')
            assert str(data['noise_sha256']) == c.array_sha(noise[start:stop])
            np.testing.assert_array_equal(data['labels'], labels[start:stop])
            assert data['pixels'].shape == (stop-start,256,256,3)
            assert data['pixels'].dtype == np.uint8 and np.isfinite(data['latents']).all()
            images.append(data['pixels']); latents.append(data['latents'])
            seconds.append(float(data['seconds'])); full.append(int(data['full'])); prefix.append(int(data['prefix']))
        records.append(dict(path=str(path),sha256=c.sha(path)))
    pixels=np.concatenate(images); zs=np.concatenate(latents)
    assert len(pixels)==request['samples'] and len(set(full))==1 and len(set(prefix))==1
    npz(out/'samples.npz',arr_0=pixels)
    npz(out/'endpoints.npz',latents=zs,labels=labels)
    gallery(pixels,out/'grid.png',labels)
    summary=dict(arm=config['arm'],config=config,complete=True,samples=len(pixels),
                 seconds=sum(seconds),full_calls_per_output=full[0],prefix_calls_per_output=prefix[0],
                 saturation_fraction=float(((pixels==0)|(pixels==255)).mean()),
                 latent_rms=float(np.sqrt(np.square(zs).mean())),
                 request_sha256=c.sha(root/'request.json'),samples_sha256=c.sha(out/'samples.npz'),records=records)
    c.atomic(out/'summary.json',summary)


def evaluate_one(root, config):
    out=root/config['arm']
    if (out/'fid.json').exists(): return
    command=[FID_PYTHON,str(c.WORK/'experiments/compute_adm_fid.py'),
             '--reference',str(REFERENCE),'--samples',str(out/'samples.npz'),
             '--batch-size','32','--gpu-memory-fraction','.3','--output',str(out/'fid.json'),
             '--activations-output',str(out/'inception_activations.npz')]
    begin=time.perf_counter()
    with (out/'fid.log').open('w') as log:
        subprocess.run(command,cwd=c.WORK,check=True,stdout=log,stderr=subprocess.STDOUT,
                       env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    c.atomic(out/'evaluation_time.json',dict(seconds=time.perf_counter()-begin))
    print(json.dumps(dict(arm=config['arm'],fid=c.read(out/'fid.json')['fid'])),flush=True)


def report(root):
    request=c.read(root/'request.json'); rows=[]
    for config in request['configs']:
        out=root/config['arm']
        if not (out/'summary.json').exists(): continue
        s=c.read(out/'summary.json')
        row={k:s[k] for k in ('arm','samples','seconds','full_calls_per_output','prefix_calls_per_output',
                              'saturation_fraction','latent_rms')}
        row.update(config)
        if (out/'fid.json').exists(): row.update(c.read(out/'fid.json'))
        rows.append(row)
    fields=['arm','kind','alpha','steps','samples','fid','sfid','inception_score',
            'seconds','full_calls_per_output','prefix_calls_per_output','saturation_fraction','latent_rms']
    with (root/'results.tmp').open('w',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    (root/'results.tmp').replace(root/'results.csv')
    c.atomic(root/'status.json',dict(completed_sampling=len(rows),completed_fid=sum('fid' in x for x in rows),
                                   planned=len(request['configs']),samples_per_arm=request['samples']))


@torch.inference_mode()
def worker(args):
    root=ROOT/args.phase;request=verify(root)
    module=importlib.import_module(request['sampler'])
    with np.load(root/'inputs.npz') as data: noise=data['noise'];labels=data['labels']
    rt=c.runtime('sit_small')
    for config in request['configs'][args.rank::args.ranks]:
        out=root/config['arm'];out.mkdir(parents=True,exist_ok=True)
        if not (out/'summary.json').exists():
            # Warmup excluded from reported sampling cost.
            module.sample(rt,c.cuda(noise[:2]),c.cuda(labels[:2]),config,snapshots=False)
            torch.cuda.synchronize()
            for start in range(0,request['samples'],request['batch']):
                path=out/f'batch{start:06d}.npz'
                if path.exists(): continue
                stop=min(start+request['batch'],request['samples'])
                torch.cuda.synchronize();begin=time.perf_counter()
                result=module.sample(rt,c.cuda(noise[start:stop]),c.cuda(labels[start:stop]),config,
                                     snapshots=start==0)
                pixels=rt.decode(result['latents'])
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                extra={}
                if 'trace' in result: extra['trace']=np.asarray(result['trace'])
                npz(path,pixels=pixels,latents=result['latents'].float().cpu().numpy(),labels=labels[start:stop],
                    seconds=seconds,full=result['counts']['full'],prefix=result['counts']['prefix'],start=start,
                    noise_sha256=c.array_sha(noise[start:stop]),request_sha256=c.sha(root/'request.json'),**extra)
                if start==0 and 'snapshots' in result:
                    snap=result['snapshots']
                    npz(out/'snapshots.npz',**{k:(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else np.asarray(v))
                                             for k,v in snap.items()})
                c.atomic(root/f'progress{args.rank}.json',dict(pid=os.getpid(),arm=config['arm'],completed=stop,
                                                           samples=request['samples'],phase='sampling'))
                del result
            collect(root,config,request,noise,labels)
        torch.cuda.empty_cache()
        c.atomic(root/f'progress{args.rank}.json',dict(pid=os.getpid(),arm=config['arm'],phase='evaluation'))
        evaluate_one(root,config)
        print('completed',config['arm'],flush=True)
    c.atomic(root/f'worker{args.rank}_complete.json',dict(complete=True,pid=os.getpid()))


def controller(args):
    root=ROOT/args.phase;verify(root);gpus=args.gpus.split(','); processes=[];logs=[]
    for rank,gpu in enumerate(gpus):
        log=(root/f'worker{rank}.log').open('a');logs.append(log)
        command=[c.PYTHON,'-m',MODULE,'worker','--phase',args.phase,'--rank',str(rank),'--ranks',str(len(gpus))]
        p=subprocess.Popen(command,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT,
                           env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        processes.append(p)
    c.atomic(root/'controller.json',dict(pid=os.getpid(),workers=[p.pid for p in processes],gpus=gpus))
    while any(p.poll() is None for p in processes):
        report(root);time.sleep(10)
    report(root);codes=[p.returncode for p in processes]
    c.atomic(root/'controller_complete.json',dict(complete=all(x==0 for x in codes),exit_codes=codes))
    for log in logs: log.close()
    if any(codes): raise RuntimeError(codes)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','controller','report'])
    p.add_argument('--phase',required=True);p.add_argument('--config');p.add_argument('--sampler')
    p.add_argument('--samples',type=int,default=1000);p.add_argument('--seed',type=int,default=2026091397)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--ranks',type=int,default=3);p.add_argument('--gpus',default='1,2,3')
    args=p.parse_args()
    if args.action=='prepare': prepare(args)
    elif args.action=='worker':worker(args)
    elif args.action=='controller':controller(args)
    else:report(ROOT/args.phase)


if __name__=='__main__':main()
