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
from . import catalog as c, core, data, train, models, checks
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha

MODULE='experiments.sit_reference_compilation_20260912.pipeline'
REPORT=c.WORK/'docs/SIT_REFERENCE_COMPILATION_RESULTS_20260912_ZH.md'
PARENT_VERIFY=engine.verify_parent


def sources():
    return sorted(set([*data.sources(),Path(engine.__file__),Path(analysis.__file__),
        Path(core.previous.__file__),Path(core.previous.previous.__file__),Path(core.previous.previous.angular.__file__)]))


def verify_parent():
    parent=PARENT_VERIFY();request=train.verify();assets=dict(parent['assets'])
    assets.update(request['assets']);assets[str(c.ROOT/'training_request.json')]=sha(c.ROOT/'training_request.json')
    fingerprints=[]
    for method in c.METHODS:
        folder=c.ROOT/'training'/method;done=read(folder/'complete.json')
        assert done['passed'] and done['step']==c.STEPS and done['strong_unchanged']
        assert done['request_sha256']==sha(c.ROOT/'training_request.json') and done['checkpoint_sha256']==sha(folder/'model.pt')
        for name in ('complete.json','model.pt','input_fingerprints.json'):assets[str(folder/name)]=sha(folder/name)
        if method.startswith('ig_'):fingerprints.append(done['input_fingerprints_sha256'])
    assert len(set(fingerprints))==1
    for method in ('ig_ig','cfg_cfg'):
        folder=c.parent.head_folder(method);receipt=read(folder/'complete.json')
        assert receipt['checkpoint_sha256']==sha(folder/'model.pt')
        for name in ('complete.json','model.pt'):assets[str(folder/name)]=sha(folder/name)
    assets[str(c.ROOT/'data/codebook.pt')]=sha(c.ROOT/'data/codebook.pt')
    return dict(parent,assets=assets)


def prepare_atomic(path,value):
    # Metadata is finalized as part of request preparation, before any sampling.
    if Path(path).name=='request.json' and value.get('stage')==c.CONFIRM:
        value=dict(value,independent_confirmation=True)
        value['references']=dict(value['references'])
        for p in (c.ROOT/c.STAGE/'request.json',c.ROOT/c.STAGE/'results.json',c.ROOT/'screen_review.json'):
            value['references'][str(p)]=sha(p)
        assert value['bank']['noise_sha256']!=read(c.ROOT/c.STAGE/'request.json')['bank']['noise_sha256']
    atomic(path,value)


def configure():
    engine.ROOT,engine.MODULE,engine.PROTOCOL=c.ROOT,MODULE,c.PROTOCOL
    engine.STAGES={c.STAGE:(1000,c.NOISE_SEED),c.CONFIRM:(5000,c.CONFIRM_SEED)}
    engine.fusion,engine.additional_sources,engine.verify_parent=core,sources,verify_parent
    engine.atomic=prepare_atomic;analysis.write_progress=write_progress


def write_progress(base,request,rows):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['arm','source','role','fid',
        'full_calls_per_image','prefix_calls_per_image','sum_batch_gpu_seconds','complete'],extrasaction='ignore')
    writer.writeheader();writer.writerows(rows);(base/'results.csv').write_text(stream.getvalue())
    portable=c.WORK/'docs/data/reference_compilation_20260912';portable.mkdir(parents=True,exist_ok=True)
    (portable/(request['stage']+'.csv')).write_text(stream.getvalue())
    lines=['# 共享参考蒸馏与候选概率 CFG：生成结果\n',
        f'阶段 {request["stage"]}，已完成{len(rows)}/{len(request["configs"])}组，每组{request["samples"]}图。\n',
        '|方法|FID↓|sFID↓|IS↑|Full/prefix|采样与解码GPU秒|','|---|--:|--:|--:|--:|--:|']
    for r in rows:
        if not r['complete']:lines.append(f'|{r["arm"]}|数值失败|—|—|—|—|');continue
        lines.append(f'|{r["arm"]}|{r["fid"]:.6f}|{r["metrics"]["sfid"]:.6f}|{r["metrics"]["inception_score"]:.4f}|'
            f'{r["full_calls_per_image"]:.0f}/{r["prefix_calls_per_image"]:.0f}|{r["sum_batch_gpu_seconds"]:.2f}|')
    lines += ['\n新1K使用全新配对噪声；5K仅在预设门槛通过后执行。IG教师独立前缀只用于离线监督，部署采样不调用它。\n',
        '[冻结协议](SIT_REFERENCE_COMPILATION_PROTOCOL_20260912_ZH.md)\n',f'质量请求SHA256：{sha(base/"request.json")}\n']
    result='\n'.join(lines);(base/'report.md').write_text(result);REPORT.write_text(result)


@torch.inference_mode()
def check_rank(rank):
    configure();verify_parent();core.install();rt=engine.parent.operators.make_runtime()
    noise=torch.from_numpy(np.load(c.OLD/'inputs/noise.npy')[:8].copy()).cuda()
    labels=torch.from_numpy(np.load(c.OLD/'inputs/labels.npy')[:8].copy()).cuda()
    configs=c.configurations();anchor=next(r for r in configs if r['arm']=='strong_00')
    native,_=core.sample(rt,noise[:2],labels[:2],anchor)
    hooks=engine.parent.hook_counts(rt);strong_hash=train.state_sha(rt.model)
    rt.labels=labels[:2]
    with models.Capture(rt) as capture:
        strong=rt.field(noise[:2],noise.new_tensor(.25),'full')
        copy_head=models.make_head(rt,'ig_deep')
        copied=models.unpatchify(rt,copy_head(capture.values[12],capture.context))
        torch.testing.assert_close(copied,strong,rtol=2e-4,atol=2e-4)
        channel_error=float((copied-strong).abs().max())
        roundtrip=models.unpatchify(rt,models.patchify(noise[:2]));assert torch.equal(roundtrip,noise[:2])
    records=[]
    for config in configs[rank::4]:
        value,stats=core.sample(rt,noise[:2],labels[:2],config)
        zero,_=core.sample(rt,noise[:2],labels[:2],config,zero=True)
        assert torch.isfinite(value).all() and torch.equal(zero,native)
        assert engine.parent.hook_counts(rt)==hooks
        assert not hasattr(rt,'trained_weak_prefixes')
        records.append(dict(arm=config['arm'],full=stats['full_calls'],prefix=stats['prefix_calls'],
            zero_exact=True,max_abs=float(value.abs().max())))
    golden=next(r for r in configs if r['arm']==('strong_00','ig_local_00','cfg_native_04','cfg_apg_07')[rank])
    value,_=core.sample(rt,noise,labels,golden)
    with np.load(c.OLD/golden['arm']/'rank0/batch0000.npz') as batch:
        np.testing.assert_array_equal(value.cpu().numpy(),batch['latents'])
    again,_=core.sample(rt,noise[:2],labels[:2],anchor)
    assert torch.equal(native,again) and train.state_sha(rt.model)==strong_hash
    atomic(c.ROOT/'development'/f'rank{rank}.json',dict(passed=True,records=records,
        source_hashes={str(p):sha(p) for p in sources()},strong_unchanged=True,old_golden_exact=True,
        native_deep_velocity_channel_error=channel_error,independent_prefix_never_loaded=True))


def development_check():
    configure();folder=c.ROOT/'development';folder.mkdir(exist_ok=True)
    atomic(c.ROOT/'analytic_checks.json',checks.analytic_checks())
    processes=[];streams=[]
    try:
        for rank in range(4):
            stream=(folder/f'rank{rank}.log').open('a');streams.append(stream)
            processes.append(subprocess.Popen([engine.PYTHON,'-u','-m',MODULE,'--check-rank',str(rank)],
                cwd=c.WORK,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('preflight failed')
            time.sleep(2)
        assert all(p.returncode==0 for p in processes)
        ranks=[read(folder/f'rank{r}.json') for r in range(4)]
        expected={str(p):sha(p) for p in sources()}
        assert all(r['passed'] and r['source_hashes']==expected for r in ranks)
        records=[v for r in ranks for v in r['records']]
        assert sorted(r['arm'] for r in records)==sorted(r['arm'] for r in c.configurations())
        atomic(c.ROOT/'development_check.json',dict(passed=True,source_hashes=expected,
            trajectories=records,old_golden_exact=True,all_zero_exact=True,independent_prefix_never_loaded=True))
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=30)
        for stream in streams:stream.close()


def run_and_audit(stage,configs,selection=None):
    engine.prepare_stage(stage,configs,selection=selection);engine.run_stage(stage)
    assert read(c.ROOT/stage/'status.json')['phase']=='complete'
    rows=read(c.ROOT/stage/'results.json')
    analysis.audit_stage(c.ROOT/stage,[r['arm'] for r in rows if r['complete']])
    write_progress(c.ROOT/stage,read(c.ROOT/stage/'request.json'),rows)
    return rows


def pipeline():
    configure();verify_parent()
    with (c.ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (c.ROOT/'STOP_AFTER_CURRENT').exists():return
        if not (c.ROOT/'development_check.json').exists():development_check()
        rows=run_and_audit(c.STAGE,c.planned());by={r['arm']:r for r in rows}
        specifications={
            'ig':dict(candidates=['ig_shallow_00','ig_deep_00'],controls=['ig_original_00','ig_ig_00','ig_adg_reference_00']),
            'cfg':dict(candidates=['cfg_probability_00'],controls=['cfg_mean_00','cfg_original_00','cfg_cfg_00','cfg_apg_07'])}
        eligible=[];confirm_arms=[]
        for source,spec in specifications.items():
            passed=[a for a in spec['candidates'] if by[a]['complete'] and all(by[b]['complete'] and by[a]['fid']<=by[b]['fid']-.5 for b in spec['controls'])]
            if passed:
                best=min(passed,key=lambda a:by[a]['fid']);eligible.append(best);confirm_arms.extend([best,*spec['controls']])
        review=dict(eligible=eligible,threshold=.5,comparisons=specifications,
            request_sha256=sha(c.ROOT/c.STAGE/'request.json'),results_sha256=sha(c.ROOT/c.STAGE/'results.json'))
        atomic(c.ROOT/'screen_review.json',review)
        confirmed=[]
        if eligible and not (c.ROOT/'STOP_AFTER_CURRENT').exists():
            configs=[r for r in c.planned() if r['arm'] in confirm_arms]
            confirmation=run_and_audit(c.CONFIRM,configs,selection=review);by5={r['arm']:r for r in confirmation}
            for arm in eligible:
                controls=specifications[by[arm]['source']]['controls']
                if by5[arm]['complete'] and all(by5[b]['complete'] and by5[arm]['fid']<=.99*by5[b]['fid'] for b in controls):confirmed.append(arm)
        atomic(c.ROOT/'status.json',dict(phase='complete',total_screen_arms=len(rows),eligible=eligible,
            confirmed=confirmed,final_audit_passed=True,numerical_failures=sum(not r['complete'] for r in rows)))


if __name__=='__main__':
    configure();p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--pipeline',action='store_true');g.add_argument('--check',action='store_true')
    g.add_argument('--check-rank',type=int);g.add_argument('--worker',type=int)
    p.add_argument('--stage');p.add_argument('--run-id');p.add_argument('--parent-pid',type=int);a=p.parse_args()
    if a.pipeline:pipeline()
    elif a.check:development_check()
    elif a.check_rank is not None:check_rank(a.check_rank)
    else:engine.worker(a.stage,a.worker,a.run_id,a.parent_pid)
