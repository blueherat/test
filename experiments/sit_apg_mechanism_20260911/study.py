"""Held-out mechanism checks before designing additional 1K FID arms.

Four ranks, 32 independent seeds, three CFG amounts, five visited times.
No FID, rejection sampling, fitting to the 1K sweep, or weight training.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from experiments.sit_apg_mechanism_20260911 import common as ops
from experiments.sit_guidance_portfolio_20260910 import operators as old
from experiments.sit_fsg_pasted_20260910 import core as semantics
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments.lifting_scale_sweep_20260909 import EXPS, WORK, atomic, read, sha, array_sha

ROOT = EXPS/'sit_apg_mechanism_extension_20260911'
STUDY = ROOT/'mechanism'
DEPENDENCY = EXPS/'sit_control_output_50ideas_20260910'
MODULE = 'experiments.sit_apg_mechanism_20260911.study'
STEPS = (8, 16, 24, 32, 40)
AMOUNTS = (0., 1.25, 2.75)
FIELDS = ('conditional', 'null', 'gap', 'orthogonal', 'cfg', 'projected',
          'fixed_cfg_a0', 'fixed_cfg_a125', 'fixed_cfg_a275')
N_SAMPLES, BATCH, N_PROBES = 32, 4, 4
SEED = 202611101


def sources():
    return [Path(__file__).resolve(), Path(ops.__file__).resolve()]


def prepare():
    STUDY.mkdir(parents=True, exist_ok=True)
    path = STUDY/'request.json'
    if path.exists():
        return verify()
    parent = engine.verify_parent()
    dependency = read(DEPENDENCY/'control_screen_1k/request.json')
    for category in ('sources', 'assets'):
        for p, digest in dependency[category].items():
            assert sha(p) == digest, p
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal((N_SAMPLES, 4, 32, 32)).astype(np.float32)
    labels = rng.permutation(100)[:N_SAMPLES].astype(np.int64)
    assert array_sha(noise) != dependency['bank']['noise_sha256']
    np.save(STUDY/'noise.npy', noise)
    np.save(STUDY/'labels.npy', labels)
    request = dict(samples=N_SAMPLES, batch=BATCH, ranks=4, steps=STEPS,
        amounts=AMOUNTS, probes=N_PROBES, seed=SEED, fields=FIELDS,
        source_hashes={**dependency['sources'], **{str(p):sha(p) for p in sources()}},
        assets={**parent['assets'], **{str(p):sha(p) for p in (semantics.MANIFEST, semantics.CLASSIFIER_WEIGHTS)}},
        dependency_request=str(DEPENDENCY/'control_screen_1k/request.json'),
        dependency_request_sha256=sha(DEPENDENCY/'control_screen_1k/request.json'),
        bank_files={name:sha(STUDY/name) for name in ('noise.npy', 'labels.npy')},
        noise_sha256=array_sha(noise), label_sha256=array_sha(labels),
        no_fid=True, no_weight_training=True, projection_is_memory_free=True,
        finite_difference_relative_radius=.001, component_pulse_steps=2,
        future_grids=(8,16), local_step_subdivisions=(1,2,4), prepared_unix=time.time())
    atomic(path, request)
    return request


def verify():
    request = read(STUDY/'request.json')
    for category in ('source_hashes', 'assets'):
        for p, digest in request[category].items():
            assert sha(p) == digest, p
    for name, digest in request['bank_files'].items():
        assert sha(STUDY/name) == digest, name
    return request


def append_rows(out, metadata, values, size):
    arrays = {key:ops.numpy(value) if isinstance(value, torch.Tensor) else value for key,value in values.items()}
    for j in range(size):
        row = {**metadata, 'sample_id':metadata['start']+j}
        for key, value in arrays.items():
            item = value[j] if isinstance(value, np.ndarray) else value
            row[key] = item.item() if isinstance(item, np.generic) else item
        out.append(row)


def differential(rt, state, t, labels, amount, generator, meta, rows):
    epsilon = .001*ops.norm(state).clamp_min(64.)
    dim = state[0].numel()
    for probe in range(N_PROBES):
        p, q = ops.random_unit(state, generator), ops.random_unit(state, generator)
        plus_p = ops.bundle(rt, state+epsilon*p, t, labels, amount)
        minus_p = ops.bundle(rt, state-epsilon*p, t, labels, amount)
        plus_q = ops.bundle(rt, state+epsilon*q, t, labels, amount)
        minus_q = ops.bundle(rt, state-epsilon*q, t, labels, amount)
        for name in FIELDS:
            jp = (plus_p[name]-minus_p[name])/(2*epsilon)
            jq = (plus_q[name]-minus_q[name])/(2*epsilon)
            anti = ops.inner(p,jq)-ops.inner(q,jp)
            append_rows(rows,dict(meta,probe=probe,field=name),dict(
                epsilon=epsilon.flatten(),
                antisymmetric_frobenius_sq=dim**2*anti.flatten().double().square(),
                jacobian_frobenius_sq=dim*(jp.double().flatten(1).square().sum(1)+jq.double().flatten(1).square().sum(1))/2,
                symmetric_rayleigh=(ops.inner(p,jp)+ops.inner(q,jq)).flatten()/2,
                circulation_bilinear=anti.flatten()),len(state))
        # One central derivative is repeated at twice the radius, providing a
        # numerical-scale check without treating every finite difference as exact.
        if probe == 0:
            high = ops.bundle(rt,state+2*epsilon*q,t,labels,amount)
            low = ops.bundle(rt,state-2*epsilon*q,t,labels,amount)
            j1 = (plus_q['gap']-minus_q['gap'])/(2*epsilon)
            j2 = (high['gap']-low['gap'])/(4*epsilon)
            append_rows(rows,dict(meta,probe=-1,field='gap_fd_scale_check'),dict(
                derivative_relative_disagreement=(ops.norm(j1-j2)/ops.norm(j2).clamp_min(1e-8)).flatten()),len(state))


def snapshot(rt, state, t, labels, amount, generator, meta, curl_rows, effect_rows):
    bundle = ops.bundle(rt,state,t,labels,amount)
    differential(rt,state,t,labels,amount,generator,meta,curl_rows)
    pulse = (2/64)*max(amount,.5)*ops.norm(bundle['gap'])
    candidates = {'zero':state,
                  'parallel':state+pulse*ops.unit(bundle['parallel']),
                  'orthogonal':state+pulse*ops.unit(bundle['orthogonal']),
                  'gap':state+pulse*ops.unit(bundle['gap'])}
    saved = {'state':ops.numpy(state), 'labels':ops.numpy(labels),
             'sample_ids':np.arange(meta['start'],meta['start']+len(state)),
             'time':t, 'amount':amount, 'pulse_radius':ops.numpy(pulse.flatten())}
    readouts = {}
    for steps in (8,16):
        for name,candidate in candidates.items():
            end = ops.future(rt,candidate,t,labels,steps)
            result = ops.readout(rt,end,labels)
            readouts[steps,name] = result
            saved[f'end{steps}_{name}'] = ops.numpy(end)
            base = readouts[steps,'zero']
            append_rows(effect_rows,dict(meta,kind='component',grid=steps,variant=name),dict(
                q=result['q'],margin=result['margin'],top1=result['target_top1'],
                q_gain=result['q']-base['q'],energy=result['energy'],
                energy_change=result['energy']-base['energy'],
                moments_change=(result['moments']-base['moments']).square().mean(1).sqrt(),
                gap_norm=ops.norm(bundle['gap']).flatten(),
                clean_gap_norm=((1-t)*ops.norm(bundle['gap'])).flatten(),
                parallel_fraction=(ops.norm(bundle['parallel'])/ops.norm(bundle['gap']).clamp_min(1e-8)).flatten()),len(state))
    # Fixed physical interval. Compare coarse and refined steps to four substeps.
    h = 1/64
    for kind in ('cfg','projected'):
        steps = {n:ops.heun(rt,state,t,h,labels,amount,kind=kind,substeps=n) for n in (1,2,4)}
        ref = steps[4]
        for n in (1,2):
            err = steps[n]-ref
            append_rows(effect_rows,dict(meta,kind='integration',grid=n,variant=kind),dict(
                local_error_rms=err.square().flatten(1).mean(1).sqrt(),
                local_error_over_step=(ops.norm(err)/(h*ops.norm(bundle[kind])).clamp_min(1e-8)).flatten(),
                excess_state_energy=(steps[n].square()-ref.square()).flatten(1).mean(1)),len(state))
        for n in (1,4):
            terminal = ops.future(rt,steps[n],t+h,labels,16)
            result = ops.readout(rt,terminal,labels)
            saved[f'after_{kind}_{n}'] = ops.numpy(terminal)
            append_rows(effect_rows,dict(meta,kind='integration_future',grid=n,variant=kind),dict(
                q=result['q'],margin=result['margin'],energy=result['energy']),len(state))
    # A matched-grid null continuation exposes passive numerical drift. Comparing
    # conditional continuation against this reference isolates a control effect.
    null_step = ops.heun(rt,state,t,h,labels,kind='null')
    null_end = ops.future(rt,null_step,t+h,labels,16)
    null_readout = ops.readout(rt,null_end,labels)
    append_rows(effect_rows,dict(meta,kind='passive_grid_defect',grid=16,variant='null'),dict(
        q=null_readout['q'],margin=null_readout['margin'],
        q_drift=null_readout['q']-readouts[16,'zero']['q']),len(state))
    saved['after_null'] = ops.numpy(null_end)
    return saved


@torch.inference_mode()
def run_rank(rank):
    request = verify()
    output = STUDY/f'rank{rank}'
    output.mkdir(exist_ok=True)
    rt = old.make_runtime()
    rt.pasted_semantic = semantics.SemanticReadout(rt)
    noise = np.load(STUDY/'noise.npy')
    labels = np.load(STUDY/'labels.npy')
    curls, effects, files = [], [], []
    started = time.perf_counter()
    counts = rt.counts.copy()
    for start in range(rank*BATCH,N_SAMPLES,4*BATCH):
        n = torch.from_numpy(noise[start:start+BATCH].copy()).cuda()
        y = torch.from_numpy(labels[start:start+BATCH].copy()).cuda()
        rt.labels = y
        generator = torch.Generator(device='cuda').manual_seed(SEED+start+9001)
        for amount in AMOUNTS:
            state = n.clone()
            for k in range(64):
                if k in STEPS:
                    metadata = dict(start=start,time=k/64,amount=amount,step=k)
                    before = rt.counts['full']
                    data = snapshot(rt,state,k/64,y,amount,generator,metadata,curls,effects)
                    data['request_sha256'] = sha(STUDY/'request.json')
                    path = output/f'start{start:02d}_a{amount:g}_step{k:02d}.npz'
                    with path.open('wb') as stream:
                        np.savez(stream,**data)
                    files.append(dict(file=path.name,sha256=sha(path),**metadata))
                    atomic(output/'progress.json',dict(snapshots=len(files),last=metadata,
                        pid=os.getpid(),full_calls=rt.counts['full']-counts['full']))
                    print(json.dumps(dict(rank=rank,**metadata,snapshot_full_calls=rt.counts['full']-before)),flush=True)
                state = ops.heun(rt,state,k/64,1/64,y,amount if k<48 else 0.)
            assert torch.isfinite(state).all()
    atomic(output/'curl_rows.json',curls)
    atomic(output/'effect_rows.json',effects)
    atomic(output/'result.json',dict(passed=True,rank=rank,files=files,
        output_hashes={name:sha(output/name) for name in ('curl_rows.json','effect_rows.json')},
        curl_rows=len(curls),effect_rows=len(effects),full_calls=rt.counts['full']-counts['full'],
        decoder_images=rt.pasted_semantic.decoded_images,
        classifier_images=rt.pasted_semantic.classified_images,
        wall_seconds=time.perf_counter()-started,request_sha256=sha(STUDY/'request.json')))


def run():
    prepare()
    status = read(DEPENDENCY/'status.json')
    assert status['phase'] in ('stopped_after_current','complete'), status
    lock = (STUDY/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes, streams = [], []
    try:
        for rank in range(4):
            stream=(STUDY/f'rank{rank}.log').open('w');streams.append(stream)
            processes.append(subprocess.Popen([engine.PYTHON,'-u','-m',MODULE,'--rank',str(rank)],
                cwd=WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT))
        atomic(STUDY/'processes.json',dict(controller_pid=os.getpid(),worker_pids=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError(f'Mechanism worker failure: {[p.poll() for p in processes]}')
            time.sleep(2)
        assert all(p.returncode==0 for p in processes)
        records=[read(STUDY/f'rank{r}/result.json') for r in range(4)]
        assert all(r['passed'] for r in records)
        for rank,record in enumerate(records):
            for name,digest in record['output_hashes'].items():assert sha(STUDY/f'rank{rank}'/name)==digest
            for file in record['files']:assert sha(STUDY/f'rank{rank}'/file['file'])==file['sha256']
        assert sum(len(r['files']) for r in records)==N_SAMPLES//BATCH*len(AMOUNTS)*len(STEPS)
        atomic(STUDY/'completed.json',dict(passed=True,ranks=records,request_sha256=sha(STUDY/'request.json')))
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=30)
        for stream in streams:stream.close()
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--run',action='store_true')
    action.add_argument('--rank',type=int,choices=range(4))
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.run:run()
    else:run_rank(args.rank)


if __name__=='__main__':
    main()
