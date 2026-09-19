"""One probability-space hypothesis and a small set of direct alternatives."""
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
from . import catalog,core,theory
from .catalog import ROOT,WORK,OLD,STAGE
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import atomic,read,sha

MODULE='experiments.sit_posterior_guidance_20260912.pipeline'
PROTOCOL=WORK/'docs/SIT_POSTERIOR_GUIDANCE_PROTOCOL_20260912_ZH.md'
REPORT=WORK/'docs/SIT_POSTERIOR_GUIDANCE_RESULTS_20260912_ZH.md'
PARENT_VERIFY=engine.verify_parent


def sources():
    return sorted(set([*Path(__file__).resolve().parent.glob('*.py'),PROTOCOL,
        Path(engine.__file__),Path(analysis.__file__),Path(core.previous.__file__),
        Path(core.previous.angular.__file__)]))


def verify_parent():
    parent=PARENT_VERIFY()
    assert read(ROOT/'theory_check.json')['passed']
    return dict(parent,assets={**parent['assets'],str(ROOT/'theory_check.json'):sha(ROOT/'theory_check.json')})


def configure():
    engine.ROOT,engine.MODULE,engine.PROTOCOL=ROOT,MODULE,PROTOCOL
    engine.STAGES={STAGE:(1000,catalog.NOISE_SEED)}
    engine.fusion,engine.additional_sources,engine.verify_parent=core,sources,verify_parent
    analysis.write_progress=write_progress


def write_progress(base,request,rows):
    stream=io.StringIO()
    fields=['arm','family','role','strength','fid','full_calls_per_image',
        'prefix_calls_per_image','sum_batch_gpu_seconds','complete']
    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(rows)
    (base/'results.csv').write_text(stream.getvalue())
    portable=WORK/'docs/data/sit_posterior_guidance_20260912'
    portable.mkdir(parents=True,exist_ok=True)
    (portable/'all_results.csv').write_text(stream.getvalue())
    lines=['# 先修改概率再聚合的最小方法实验\n',
        f'已提交 {len(rows)}/{len(request["configs"])} 组配对1K，数值失败 {sum(not r["complete"] for r in rows)}。\n',
        '一个候选，额外概率对比强度固定为1；其余为原生基线和有区分度的对照。没有逐层选择、局部质量探针或强度网格。\n',
        '|方法|FID|Full/prefix调用|采样及解码GPU秒|完成|','|---|--:|--:|--:|---|']
    for r in rows:
        if not r['complete']:
            lines.append(f'|{r["family"]}|数值失败，不计算FID|—|—|False|')
            continue
        lines.append(f'|{r["family"]}|{r["fid"]:.6f}|{r["full_calls_per_image"]:.0f}/{r["prefix_calls_per_image"]:.0f}|'
            f'{r["sum_batch_gpu_seconds"]:.2f}|{r["complete"]}|')
    lines += ['\n概率候选每次前向还有12次额外QKV和logit乘法，不能把同为128次完整前向称为等算力。'
        '当前attention只作为待验证的解释概率代理；保持attention聚合的凸性不保证终点图像质量。\n',
        '[概率对象、反例与失败条件](SIT_POSTERIOR_GUIDANCE_PROTOCOL_20260912_ZH.md) · '
        '[完整逐组结果](data/sit_posterior_guidance_20260912/all_results.csv)\n',
        '完成后复盘，不自动加密参数或启动旧宽队列。\n',f'请求SHA256：{sha(base/"request.json")}。\n']
    text='\n'.join(lines)
    (base/'report.md').write_text(text);REPORT.write_text(text)


@torch.inference_mode()
def check_rank(rank):
    configure();verify_parent();core.install()
    rt=engine.parent.operators.make_runtime()
    n=torch.from_numpy(np.load(OLD/'inputs/noise.npy')[:8].copy()).cuda()
    y=torch.from_numpy(np.load(OLD/'inputs/labels.npy')[:8].copy()).cuda()
    rt.labels=y
    hooks=engine.parent.hook_counts(rt)
    records=[]
    configs=catalog.configurations()
    anchor=next(c for c in configs if c['family']=='strong')
    native,_=core.sample(rt,n[:2],y[:2],anchor)
    for c in configs[rank::4]:
        z,stats=core.sample(rt,n[:2],y[:2],c)
        assert torch.isfinite(z).all()
        zero,_=core.sample(rt,n[:2],y[:2],c,zero=True)
        assert torch.equal(zero,native),c['arm']
        assert engine.parent.hook_counts(rt)==hooks
        records.append(dict(arm=c['arm'],max_abs=float(z.abs().max()),
            full_calls=stats['full_calls'],extra_qkv=stats.get('internal_extra_qkv_calls',0),zero_exact=True))
    errors=[]
    for t in (0.,.375,.75):
        expected=core.forward(rt,n[:2],t,y[:2],0.,'manual_kernel')
        actual=core.forward(rt,n[:2],t,y[:2],0.,'manual_kernel',force_manual=True)
        relative=float((expected-actual).norm()/expected.norm().clamp_min(1e-12))
        assert relative<.003,relative
        errors.append(relative)
    golden_config=next(c for c in configs if c['arm']==
        ('strong_00','ig_local_00','cfg_native_04','cfg_apg_07')[rank])
    z,_=core.sample(rt,n,y,golden_config)
    with np.load(OLD/golden_config['arm']/'rank0/batch0000.npz') as old:
        np.testing.assert_array_equal(z.cpu().numpy(),old['latents'])
    again,_=core.sample(rt,n[:2],y[:2],anchor)
    assert torch.equal(again,native)
    atomic(ROOT/'development'/f'rank{rank}.json',dict(passed=True,rank=rank,
        records=records,manual_kernel_relative_errors=errors,old_baseline_exact=True,
        source_hashes={str(p):sha(p) for p in sources()},runtime_sources=rt.sources))


def development_check():
    configure();verify_parent()
    folder=ROOT/'development';folder.mkdir(parents=True,exist_ok=True)
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
        ranks=[read(folder/f'rank{r}.json') for r in range(4)]
        records=[x for rank in ranks for x in rank['records']]
        assert sorted(x['arm'] for x in records)==sorted(c['arm'] for c in catalog.configurations())
        expected={str(p):sha(p) for p in sources()}
        assert all(rank['passed'] and rank['source_hashes']==expected for rank in ranks)
        atomic(ROOT/'development_check.json',dict(passed=True,trajectories=len(records),
            source_hashes=expected,all_zero_strength_exact=True,four_original_baselines_exact=True,
            maximum_manual_kernel_relative_error=max(max(r['manual_kernel_relative_errors']) for r in ranks),
            no_fid_used=True))
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
        if not (ROOT/'development_check.json').exists():development_check()
        engine.prepare_stage(STAGE,catalog.configurations())
        actual=read(ROOT/STAGE/'request.json')['bank'];expected=read(OLD/'request.json')['bank']
        assert actual['noise_sha256']==expected['noise_sha256']
        assert actual['label_sha256']==expected['label_sha256']
        engine.run_stage(STAGE)
        if read(ROOT/STAGE/'status.json')['phase']!='complete':return
        rows=read(ROOT/STAGE/'results.json')
        analysis.audit_stage(ROOT/STAGE,[r['arm'] for r in rows if r['complete']])
        write_progress(ROOT/STAGE,read(ROOT/STAGE/'request.json'),rows)
        atomic(ROOT/'status.json',dict(phase='complete',total_arms=len(rows),
            numerical_failures=sum(not r['complete'] for r in rows),final_audit_passed=True))


if __name__=='__main__':
    configure()
    p=argparse.ArgumentParser();a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--pipeline',action='store_true');a.add_argument('--check',action='store_true')
    a.add_argument('--check-rank',type=int);a.add_argument('--worker',type=int)
    p.add_argument('--stage');p.add_argument('--run-id');p.add_argument('--parent-pid',type=int)
    args=p.parse_args()
    if args.pipeline:pipeline()
    elif args.check:development_check()
    elif args.check_rank is not None:check_rank(args.check_rank)
    else:engine.worker(args.stage,args.worker,args.run_id,args.parent_pid)
