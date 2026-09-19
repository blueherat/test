from __future__ import annotations
import argparse
import csv
import fcntl
import io
from . import catalog as c, core, train, models, checks
from experiments.sit_reference_compilation_20260912 import pipeline as previous
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha

engine,analysis=previous.engine,previous.analysis
PARENT_VERIFY=previous.PARENT_VERIFY
PARENT_SOURCES=previous.sources()
MODULE='experiments.sit_bayes_risk_reference_20260912.pipeline'
REPORT=c.WORK/'docs/SIT_BAYES_RISK_REFERENCE_RESULTS_20260912_ZH.md'


def sources():return sorted(set([*PARENT_SOURCES,*train.sources()]))


def verify_parent():
    parent=PARENT_VERIFY();request=train.verify();assets=dict(parent['assets'])
    assets.update(request['assets']);assets[str(c.ROOT/'training_request.json')]=sha(c.ROOT/'training_request.json')
    fingerprints=[];initial=[]
    for method in c.METHODS:
        folder=c.ROOT/'training'/method;done=read(folder/'complete.json')
        assert done['passed'] and done['step']==c.STEPS and done['strong_unchanged'] and done['strong_gradients_absent']
        assert done['request_sha256']==sha(c.ROOT/'training_request.json')
        assert done['checkpoint_sha256']==sha(folder/'model.pt')
        assert done['input_fingerprints_sha256']==sha(folder/'input_fingerprints.json')
        for name in ('complete.json','model.pt','input_fingerprints.json'):assets[str(folder/name)]=sha(folder/name)
        fingerprints.append(done['input_fingerprints_sha256']);initial.append(done['initial_head_sha256'])
        assert not done['teacher_supervision']
    assert len(set(fingerprints))==1 and len(set(initial))==1
    for method in ('ig_ig','cfg_cfg'):
        folder=c.parent.head_folder(method);receipt=read(folder/'complete.json')
        assert receipt['checkpoint_sha256']==sha(folder/'model.pt')
        for name in ('complete.json','model.pt'):assets[str(folder/name)]=sha(folder/name)
    return dict(parent,assets=assets)


def write_progress(base,request,rows):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['arm','source','role','fid',
        'full_calls_per_image','prefix_calls_per_image','sum_batch_gpu_seconds','complete'],extrasaction='ignore')
    writer.writeheader();writer.writerows(rows);(base/'results.csv').write_text(stream.getvalue())
    portable=c.WORK/'docs/data/bayes_risk_reference_20260912';portable.mkdir(parents=True,exist_ok=True)
    (portable/(request['stage']+'.csv')).write_text(stream.getvalue())
    lines=['# 由损失定义的 IG 后验弱参考\n',
        f'阶段 {request["stage"]}，已完成{len(rows)}/{len(request["configs"])}组，每组{request["samples"]}图。\n',
        '|方法|FID↓|sFID↓|IS↑|Full/prefix|采样与解码GPU秒|','|---|--:|--:|--:|--:|--:|']
    for r in rows:
        if not r['complete']:lines.append(f'|{r["arm"]}|数值失败|—|—|—|—|');continue
        lines.append(f'|{r["arm"]}|{r["fid"]:.6f}|{r["metrics"]["sfid"]:.6f}|{r["metrics"]["inception_score"]:.4f}|'
            f'{r["full_calls_per_image"]:.0f}/{r["prefix_calls_per_image"]:.0f}|{r["sum_batch_gpu_seconds"]:.2f}|')
    lines+=['\n两个新头初始化、输入和训练步数相同，仅逐坐标训练损失不同。部署均为共享第四层读出，无教师。\n',
        '[冻结协议](SIT_BAYES_RISK_REFERENCE_PROTOCOL_20260912_ZH.md)\n',f'质量请求SHA256：{sha(base/"request.json")}\n']
    result='\n'.join(lines);(base/'report.md').write_text(result);REPORT.write_text(result)


def configure():
    # Reuse checked orchestration in this process without modifying frozen files.
    previous.c,previous.core,previous.train,previous.models,previous.checks=c,core,train,models,checks
    previous.MODULE,previous.REPORT=MODULE,REPORT
    previous.sources,previous.verify_parent,previous.write_progress=sources,verify_parent,write_progress
    previous.configure()


def pipeline():
    configure();verify_parent()
    with (c.ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (c.ROOT/'STOP_AFTER_CURRENT').exists():return
        if not (c.ROOT/'development_check.json').exists():previous.development_check()
        rows=previous.run_and_audit(c.STAGE,c.planned());by={r['arm']:r for r in rows}
        candidate='ig_quartic_00'
        controls=[r['arm'] for r in c.planned() if r['role']=='control']
        eligible=[candidate] if by[candidate]['complete'] and all(by[b]['complete'] and by[candidate]['fid']<=by[b]['fid']-.5 for b in controls) else []
        review=dict(eligible=eligible,threshold=.5,candidate=candidate,controls=controls,
            request_sha256=sha(c.ROOT/c.STAGE/'request.json'),results_sha256=sha(c.ROOT/c.STAGE/'results.json'))
        atomic(c.ROOT/'screen_review.json',review);confirmed=[]
        if eligible and not (c.ROOT/'STOP_AFTER_CURRENT').exists():
            confirmation=previous.run_and_audit(c.CONFIRM,c.planned(),selection=review);by5={r['arm']:r for r in confirmation}
            if by5[candidate]['complete'] and all(by5[b]['complete'] and by5[candidate]['fid']<=.99*by5[b]['fid'] for b in controls):confirmed.append(candidate)
        atomic(c.ROOT/'status.json',dict(phase='complete',total_screen_arms=len(rows),eligible=eligible,
            confirmed=confirmed,final_audit_passed=True,numerical_failures=sum(not r['complete'] for r in rows)))



if __name__=='__main__':
    configure();p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--pipeline',action='store_true');g.add_argument('--check',action='store_true')
    g.add_argument('--check-rank',type=int);g.add_argument('--worker',type=int)
    p.add_argument('--stage');p.add_argument('--run-id');p.add_argument('--parent-pid',type=int);a=p.parse_args()
    if a.pipeline:pipeline()
    elif a.check:previous.development_check()
    elif a.check_rank is not None:previous.check_rank(a.check_rank)
    else:engine.worker(a.stage,a.worker,a.run_id,a.parent_pid)
