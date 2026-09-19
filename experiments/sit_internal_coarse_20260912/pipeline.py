"""Train two matched readouts, then run three fixed-strength paired 1K arms."""
from __future__ import annotations
import argparse
import csv
import fcntl
import io
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import catalog,core,train
from .catalog import ROOT,WORK,OLD,STAGE,PROTOCOL
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import atomic,read,sha

MODULE='experiments.sit_internal_coarse_20260912.pipeline'
REPORT=WORK/'docs/SIT_INTERNAL_COARSE_RESULTS_20260912_ZH.md'
PARENT_VERIFY=engine.verify_parent


def sources():
    return sorted(set([*Path(__file__).resolve().parent.glob('*.py'),PROTOCOL,
        Path(engine.__file__),Path(analysis.__file__),Path(core.previous.__file__),
        Path(core.previous.angular.__file__),*train.training_sources()]))


def verify_parent():
    parent=PARENT_VERIFY();train.verify()
    assets=dict(parent['assets'])
    assets[str(ROOT/'training_request.json')]=sha(ROOT/'training_request.json')
    for method in catalog.METHODS:
        folder=ROOT/'training'/method;done=read(folder/'complete.json')
        assert done['passed'] and done['step']==catalog.STEPS
        assert done['request_sha256']==sha(ROOT/'training_request.json')
        assert done['checkpoint_sha256']==sha(folder/'model.pt')
        for name in ('complete.json','model.pt'):assets[str(folder/name)]=sha(folder/name)
    return dict(parent,assets=assets)


def configure():
    engine.ROOT,engine.MODULE,engine.PROTOCOL=ROOT,MODULE,PROTOCOL
    engine.STAGES={STAGE:(1000,catalog.NOISE_SEED)}
    engine.fusion,engine.additional_sources,engine.verify_parent=core,sources,verify_parent
    analysis.write_progress=write_progress


def write_progress(base,request,rows):
    stream=io.StringIO()
    fields=['arm','family','role','strength','fid','full_calls_per_image',
        'prefix_calls_per_image','sum_batch_gpu_seconds','complete']
    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (base/'results.csv').write_text(stream.getvalue())
    portable=WORK/'docs/data/sit_internal_coarse_20260912';portable.mkdir(parents=True,exist_ok=True)
    (portable/'all_results.csv').write_text(stream.getvalue())
    lines=['# 高阶粗化内部预测头：固定强度实验\n',
        f'已完成 {len(rows)}/{len(request["configs"])} 组配对1K。强度0.8、深度4、原IG时间区间全部固定。\n',
        '|内部预测头|FID|Full/prefix调用|采样及解码GPU秒|完成|','|---|--:|--:|--:|---|']
    for r in rows:
        if not r['complete']:
            lines.append(f'|{r["family"]}|数值失败，不计算FID|—|—|False|')
            continue
        lines.append(f'|{r["family"]}|{r["fid"]:.6f}|{r["full_calls_per_image"]:.0f}/{r["prefix_calls_per_image"]:.0f}|'
            f'{r["sum_batch_gpu_seconds"]:.2f}|{r["complete"]}|')
    lines+=['\n两个新头均在同一原50K头上短训1500步；训练开销另报，不混入采样NFE。\n']
    for method in catalog.METHODS:
        done=read(ROOT/'training'/method/'complete.json')
        lines.append(f'- {method}：{done["training_seconds"]:.2f} GPU秒，{done["trainable_parameters"]} 个训练参数。')
    lines+=['\n历史同bank局部IG为63.202383，但需要额外64次prefix；当前三组不使用局部化。'
        '若CLT只胜过短训native、不胜过原头，不能认领IG改进。\n',
        '[理论、训练约束和失败条件](SIT_INTERNAL_COARSE_PROTOCOL_20260912_ZH.md) · '
        '[完整逐组结果](data/sit_internal_coarse_20260912/all_results.csv)\n',
        f'请求SHA256：{sha(base/"request.json")}。\n']
    output='\n'.join(lines);(base/'report.md').write_text(output);REPORT.write_text(output)


def train_heads():
    train.prepare();folder=ROOT/'training';folder.mkdir(exist_ok=True)
    processes=[];streams=[]
    try:
        for gpu,method in enumerate(catalog.METHODS):
            log=(folder/f'{method}.log').open('a');streams.append(log)
            processes.append(subprocess.Popen([engine.PYTHON,'-u','-m',
                'experiments.sit_internal_coarse_20260912.train','--train',method],cwd=WORK,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError(f'Head training failed: {[p.poll() for p in processes]}')
            states={m:read(ROOT/'training'/m/'status.json') for m in catalog.METHODS
                if (ROOT/'training'/m/'status.json').exists()}
            atomic(ROOT/'status.json',dict(phase='training_heads',methods=states))
            time.sleep(5)
        assert all(p.returncode==0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=30)
        for log in streams:log.close()


@torch.inference_mode()
def check_rank(rank):
    configure();verify_parent();core.install()
    rt=engine.parent.operators.make_runtime()
    n=torch.from_numpy(np.load(OLD/'inputs/noise.npy')[:8].copy()).cuda()
    y=torch.from_numpy(np.load(OLD/'inputs/labels.npy')[:8].copy()).cuda()
    rt.labels=y;hooks=engine.parent.hook_counts(rt);original_head=rt.head
    configs=catalog.configurations();anchor=next(c for c in configs if c['family']=='strong')
    native,_=core.sample(rt,n[:2],y[:2],anchor);records=[]
    for c in configs[rank::4]:
        z,stats=core.sample(rt,n[:2],y[:2],c)
        zero,_=core.sample(rt,n[:2],y[:2],c,zero=True)
        assert torch.isfinite(z).all() and torch.equal(zero,native),c['arm']
        assert engine.parent.hook_counts(rt)==hooks and rt.head is original_head
        records.append(dict(arm=c['arm'],max_abs=float(z.abs().max()),full_calls=stats['full_calls'],zero_exact=True))
    golden=next(c for c in configs if c['arm']==('strong_00','ig_local_00','cfg_native_04','cfg_apg_07')[rank])
    z,_=core.sample(rt,n,y,golden)
    with np.load(OLD/golden['arm']/'rank0/batch0000.npz') as saved:
        np.testing.assert_array_equal(z.cpu().numpy(),saved['latents'])
    ordinary=catalog.planned()[0];z,_=core.sample(rt,n,y,ordinary)
    with np.load(catalog.MEASURE/'measure_screen_1k/ig_native_matched_00/rank0/batch0000.npz') as saved:
        np.testing.assert_array_equal(z.cpu().numpy(),saved['latents'])
    again,_=core.sample(rt,n[:2],y[:2],anchor)
    assert torch.equal(again,native) and rt.head is original_head
    atomic(ROOT/'development'/f'rank{rank}.json',dict(passed=True,rank=rank,records=records,
        old_baseline_exact=True,ordinary_ig_trajectory_exact=True,strong_unchanged=True,
        source_hashes={str(p):sha(p) for p in sources()},runtime_sources=rt.sources))


def development_check():
    configure();verify_parent();folder=ROOT/'development';folder.mkdir(exist_ok=True)
    processes=[];streams=[]
    try:
        for rank in range(4):
            log=(folder/f'rank{rank}.log').open('a');streams.append(log)
            processes.append(subprocess.Popen([engine.PYTHON,'-u','-m',MODULE,'--check-rank',str(rank)],
                cwd=WORK,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError(f'Preflight failed: {[p.poll() for p in processes]}')
            time.sleep(2)
        assert all(p.returncode==0 for p in processes)
        ranks=[read(folder/f'rank{r}.json') for r in range(4)]
        records=[record for rank in ranks for record in rank['records']]
        assert sorted(r['arm'] for r in records)==sorted(c['arm'] for c in catalog.configurations())
        expected={str(p):sha(p) for p in sources()}
        assert all(r['passed'] and r['source_hashes']==expected for r in ranks)
        atomic(ROOT/'development_check.json',dict(passed=True,trajectories=len(records),
            source_hashes=expected,all_zero_strength_exact=True,four_original_baselines_exact=True,
            ordinary_ig_trajectory_exact=True,no_fid_used=True))
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=30)
        for log in streams:log.close()


def pipeline():
    configure();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'STOP_AFTER_CURRENT').exists():return
        train_heads()
        if (ROOT/'STOP_AFTER_CURRENT').exists():return
        if not (ROOT/'development_check.json').exists():development_check()
        engine.prepare_stage(STAGE,catalog.planned())
        bank=read(ROOT/STAGE/'request.json')['bank'];old=read(OLD/'request.json')['bank']
        assert bank['noise_sha256']==old['noise_sha256'] and bank['label_sha256']==old['label_sha256']
        engine.run_stage(STAGE)
        if read(ROOT/STAGE/'status.json')['phase']!='complete':return
        rows=read(ROOT/STAGE/'results.json')
        analysis.audit_stage(ROOT/STAGE,[r['arm'] for r in rows if r['complete']])
        write_progress(ROOT/STAGE,read(ROOT/STAGE/'request.json'),rows)
        atomic(ROOT/'status.json',dict(phase='complete',total_arms=len(rows),
            numerical_failures=sum(not r['complete'] for r in rows),final_audit_passed=True))


if __name__=='__main__':
    configure();p=argparse.ArgumentParser();a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--pipeline',action='store_true');a.add_argument('--check',action='store_true')
    a.add_argument('--check-rank',type=int);a.add_argument('--worker',type=int)
    p.add_argument('--stage');p.add_argument('--run-id');p.add_argument('--parent-pid',type=int)
    args=p.parse_args()
    if args.pipeline:pipeline()
    elif args.check:development_check()
    elif args.check_rank is not None:check_rank(args.check_rank)
    else:engine.worker(args.stage,args.worker,args.run_id,args.parent_pid)
