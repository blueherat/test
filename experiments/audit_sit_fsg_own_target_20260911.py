"""Companion target-matched audit; never changes the frozen primary experiment."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from experiments.sit_fsg_ctrl_hypothesis_20260911 import core, pipeline as p, study
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha, WORK

ROOT=p.ROOT/'own_target_audit'
PROTOCOL=WORK/'docs/SIT_FSG_OWN_TARGET_AUDIT_PROTOCOL_20260911_ZH.md'
MODULE='experiments.audit_sit_fsg_own_target_20260911'


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),PROTOCOL,Path(core.__file__),Path(study.__file__)]
    request=dict(sources={str(x.resolve()):sha(x) for x in paths},
        study_request_sha256=sha(p.ROOT/'study_request.json'),
        sample_indices=list(range(32)),bases=['cfg_tuned','cfg_high'],times=[8,24,40],
        horizons=[2/64,4/64,8/64,16/64],batch=8,
        precision='FP32 network, TF32 matmul off; same as original Jacobian study',
        outcomes='Direct finite NULL endpoint errors against both frozen targets; no Jacobian approximation',
        variants=['euler_fsg','cfg_equal','euler_capped','cfg_capped','cfg_first_order',
                  'flow_inverse','own_euler_inverse'],created_unix=time.time())
    path=ROOT/'request.json'
    if path.exists():
        existing=read(path);request['created_unix']=existing['created_unix'];assert request==existing
    else:atomic(path,request)
    return request


def verify():
    request=read(ROOT/'request.json')
    for path,digest in request['sources'].items():assert sha(path)==digest,path
    assert request['study_request_sha256']==sha(p.ROOT/'study_request.json')
    return request


def pause_at_sampling_barrier():
    while True:
        status=read(p.ROOT/'status.json')
        if status['phase']=='complete':return None
        assert status['phase'] not in ('failed','stopped_after_current'),status
        if status['phase']!='sampling':time.sleep(.25);continue
        pid=status['controller_pid']
        assert 'sit_fsg_ctrl_hypothesis_20260911.pipeline' in Path(f'/proc/{pid}/cmdline').read_text()
        command_path=p.ROOT/p.STAGE/'runs'/status['run_id']/'command.json'
        command=read(command_path)
        if command.get('arm')!=status['arm'] or command.get('command')!='sample':
            time.sleep(.05);continue
        os.kill(pid,signal.SIGSTOP)
        try:
            while Path(f'/proc/{pid}/stat').read_text().split(')')[1].split()[0]!='T':time.sleep(.01)
            held=read(p.ROOT/'status.json');command=read(command_path)
            if held['phase']!='sampling' or held.get('arm')!=command.get('arm'):
                os.kill(pid,signal.SIGCONT);time.sleep(.1);continue
            atomic(ROOT/'pause.json',dict(status=held,paused_unix=time.time()))
            paths=[p.ROOT/p.STAGE/held['arm']/f'rank{r}/summary.json' for r in range(4)]
            while not all(x.exists() and read(x).get('run_id')==held['run_id'] for x in paths):time.sleep(.5)
            atomic(ROOT/'sampling_barrier.json',dict(reached_unix=time.time(),summaries={str(x):sha(x) for x in paths}))
            return pid
        except BaseException:
            os.kill(pid,signal.SIGCONT);raise


@torch.inference_mode()
@core.old.exact_matmul()
def worker(rank):
    request=verify();request_hash=sha(ROOT/'request.json');start=rank*8
    rt=core.old.make_runtime()
    # Runtime construction restores the established native precision policy.
    torch.backends.cuda.matmul.allow_tf32=False
    labels=torch.from_numpy(np.load(p.ROOT/'inputs/labels.npy')[start:start+8].copy()).cuda()
    before=rt.counts.copy();begin=time.perf_counter();rows=[];max_original_difference=0.
    for base,amount in (('cfg_tuned',1.25),('cfg_high',2.75)):
        for step in (8,24,40):
            x=study.load_state(base,start,step);t=step/64
            original=read(p.ROOT/'jacobian'/f'{base}_{step:02d}_{start:03d}.json')['rows']
            previous={(r['index'],r['horizon'],r['variant']):r for r in original}
            for horizon in request['horizons']:
                end=t+horizon
                function=lambda z:core.interval(rt,z,t,end,labels,null=True,steps=8)
                c,u=core.pair(rt,x,t,labels);gap=c-u
                null=function(x)
                flow_target=core.interval(rt,x,t,end,labels,amount=amount,steps=8)
                own_target=x+horizon*(c+amount*gap)
                euler=own_target-horizon*core.field(rt,own_target,end,labels,True)-x
                adapter,radius,_=core.fsg_delta(rt,x,t,labels,amount,horizon)
                torch.testing.assert_close(euler,adapter,rtol=0,atol=0)
                cap=core.cap(euler,radius)
                directions=dict(euler_fsg=euler,cfg_equal=core.norm(euler)*core.unit(gap),
                    euler_capped=cap,cfg_capped=core.norm(cap)*core.unit(gap),
                    cfg_first_order=horizon*(1+amount)*gap,
                    flow_inverse=core.interval(rt,flow_target,end,t,labels,null=True,steps=8)-x,
                    own_euler_inverse=core.interval(rt,own_target,end,t,labels,null=True,steps=8)-x)
                torch.testing.assert_close(core.norm(directions['cfg_equal']),core.norm(euler),rtol=2e-6,atol=1e-7)
                for variant,delta in directions.items():
                    endpoint=function(x+delta)
                    flow_error=study.relative(endpoint-flow_target,flow_target-null)
                    values=dict(displacement_rms=core.rms(delta),
                        effective_coefficient_per_H=(core.norm(delta)/(horizon*core.norm(gap))).flatten(),
                        first_order_length_ratio=(core.norm(delta)/(horizon*(1+amount)*core.norm(gap))).flatten(),
                        direction_cos_cfg=study.cosine(delta,gap),
                        finite_error_flow_target=flow_error,
                        finite_error_own_euler_target=study.relative(endpoint-own_target,own_target-null),
                        flow_vs_euler_target_relative_difference=study.relative(flow_target-own_target,flow_target-null))
                    added=study.encode_records(8,dict(start=start,base=base,amount=amount,w=1+amount,k=step,
                        horizon=horizon,variant=variant),values)
                    old_name={'cfg_equal':'euler_cfg_equal','cfg_capped':'capped_cfg_equal'}.get(variant,variant)
                    for row in added:
                        old=previous.get((row['index'],horizon,old_name))
                        if old is not None:
                            difference=abs(row['finite_error_flow_target']-old['finite_target_relative_error'])
                            max_original_difference=max(max_original_difference,difference)
                            assert difference<2e-5,(base,step,horizon,variant,difference)
                    rows+=added
            print(dict(rank=rank,base=base,k=step,rows=len(rows)),flush=True)
    verify()
    atomic(ROOT/f'rank{rank}.json',dict(complete=True,request_sha256=request_hash,rows=rows,
        total_full_calls=rt.counts['full']-before['full'],elapsed_seconds=time.perf_counter()-begin,
        max_replayed_flow_target_error_difference=max_original_difference))


def run(pause):
    prepare();pid=pause_at_sampling_barrier() if pause else None
    children=[];streams=[]
    try:
        for rank in range(4):
            stream=(ROOT/f'rank{rank}.log').open('a');streams.append(stream)
            children.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'--worker',str(rank)],cwd=WORK,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                stdout=stream,stderr=subprocess.STDOUT))
        while any(x.poll() is None for x in children):
            if any(x.poll() not in (None,0) for x in children):raise RuntimeError('Own-target audit worker failed')
            time.sleep(.5)
        assert all(x.returncode==0 for x in children)
        results=[read(ROOT/f'rank{r}.json') for r in range(4)]
        assert all(x['complete'] for x in results)
        assert sum(len(x['rows']) for x in results)==5376
        verify();atomic(ROOT/'complete.json',dict(complete=True,rows=5376,
            request_sha256=sha(ROOT/'request.json'),finished_unix=time.time(),
            max_replayed_difference=max(x['max_replayed_flow_target_error_difference'] for x in results)))
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:child.wait(timeout=30)
        for stream in streams:stream.close()
        if pid is not None:
            os.kill(pid,signal.SIGCONT);atomic(ROOT/'resumed.json',dict(controller_pid=pid,resumed_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pause-small',action='store_true')
    parser.add_argument('--worker',type=int,choices=range(4));args=parser.parse_args()
    if args.worker is None:run(args.pause_small)
    else:worker(args.worker)
