"""Confirm the fixed native-prefix control on independent paired 5K noise."""
from __future__ import annotations
import argparse
import csv
import fcntl
import io
from pathlib import Path
import numpy as np
from experiments.sit_prefix_coarse_20260912 import pipeline as prior
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import WORK,EXPS,atomic,read,sha

ROOT=EXPS/'sit_prefix_confirmation_20260912'
SCREEN=prior.ROOT/prior.STAGE
STAGE='prefix_confirm_5k'
MODULE='experiments.sit_prefix_confirmation_20260912'
PROTOCOL=WORK/'docs/SIT_PREFIX_CONFIRMATION_PROTOCOL_20260912_ZH.md'
REPORT=WORK/'docs/SIT_PREFIX_CONFIRMATION_RESULTS_20260912_ZH.md'
SEED=2026091205


def sources():return sorted(set([*prior.sources(),Path(__file__).resolve(),PROTOCOL]))


def selection():
    request=read(SCREEN/'request.json');rows=read(SCREEN/'results.json')
    assert read(SCREEN/'analysis_audit.json')['passed']
    names=['ig_prefix_original_00','ig_prefix_local_00','ig_prefix_native_00']
    chosen=[next(c for c in request['configs'] if c['arm']==name) for name in names]
    return dict(configs=chosen,confirmation_candidate_arm='ig_prefix_native_00',
        posthoc_control_promotion=True,no_retraining_or_hyperparameter_change=True,
        screen_request_sha256=sha(SCREEN/'request.json'),screen_results_sha256=sha(SCREEN/'results.json'),
        screen_audit_sha256=sha(SCREEN/'analysis_audit.json'),screen_fids={r['arm']:r['fid'] for r in rows},
        hypothesis='The native weak-prefix improvement over original and local IG survives an independent noise bank.',
        clt_hypothesis_rejected_for_current_construction=True,samples=5000,noise_seed=SEED)


def verify_parent():
    parent=prior.verify_parent();chosen=selection()
    assert read(ROOT/'selection.json')==chosen
    assets=dict(parent['assets'])
    for p in (ROOT/'selection.json',SCREEN/'request.json',SCREEN/'results.json',SCREEN/'analysis_audit.json'):
        assets[str(p)]=sha(p)
    return dict(parent,assets=assets)


def configure():
    engine.ROOT,engine.MODULE,engine.PROTOCOL=ROOT,MODULE,PROTOCOL
    engine.STAGES={STAGE:(5000,SEED)}
    engine.fusion,engine.additional_sources,engine.verify_parent=prior.core,sources,verify_parent
    analysis.write_progress=write_progress


def write_progress(base,request,rows):
    fields=['arm','family','role','strength','fid','full_calls_per_image',
        'prefix_calls_per_image','sum_batch_gpu_seconds','complete']
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(rows);(base/'results.csv').write_text(stream.getvalue())
    folder=WORK/'docs/data/sit_prefix_confirmation_20260912';folder.mkdir(parents=True,exist_ok=True)
    (folder/'all_results.csv').write_text(stream.getvalue())
    lines=['# 原分布弱前缀：独立5K确认\n',f'已完成{len(rows)}/3组；新噪声{SEED}，同一5K输入配对。\n',
        '|方法|FID|sFID|IS|Full/prefix|采样及解码GPU秒|','|---|--:|--:|--:|--:|--:|']
    for r in rows:
        if not r['complete']:
            lines.append(f'|{r["family"]}|数值失败|—|—|—|—|');continue
        metrics=r['metrics']
        lines.append(f'|{r["family"]}|{r["fid"]:.6f}|{metrics["sfid"]:.6f}|'
            f'{metrics["inception_score"]:.4f}|{r["full_calls_per_image"]:.0f}/{r["prefix_calls_per_image"]:.0f}|'
            f'{r["sum_batch_gpu_seconds"]:.2f}|')
    lines+=['\nnative来自前一轮对照的事后发现；当前确认不支持粗化机制，也不单独建立新方法。'
        '本轮不训练、不调整参数；5K与旧1K不能直接相减。\n',
        '[冻结协议](SIT_PREFIX_CONFIRMATION_PROTOCOL_20260912_ZH.md) · '
        '[全部结果](data/sit_prefix_confirmation_20260912/all_results.csv)\n',
        f'请求SHA256：{sha(base/"request.json")}。\n']
    output='\n'.join(lines);(base/'report.md').write_text(output);REPORT.write_text(output)


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);chosen=selection()
    if (ROOT/'selection.json').exists():assert read(ROOT/'selection.json')==chosen
    else:atomic(ROOT/'selection.json',chosen)
    verify_parent();check=read(prior.ROOT/'development_check.json');assert check['passed']
    for path,digest in check['source_hashes'].items():assert sha(path)==digest,path
    newcheck=dict(passed=True,source_hashes={str(p):sha(p) for p in sources()},
        reused_preflight=str(prior.ROOT/'development_check.json'),
        reused_preflight_sha256=sha(prior.ROOT/'development_check.json'),
        identical_sampler_and_weights=True,configs_exact_screen_subset=True,
        no_new_gpu_preflight_claimed=True)
    if (ROOT/'development_check.json').exists():assert read(ROOT/'development_check.json')==newcheck
    else:atomic(ROOT/'development_check.json',newcheck)
    base=ROOT/STAGE
    existed=(base/'request.json').exists()
    engine.prepare_stage(STAGE,chosen['configs'],selection=chosen)
    request=read(base/'request.json')
    if not existed:
        request['independent_confirmation']=True
        request['confirmation_candidate_arm']=chosen['confirmation_candidate_arm']
        atomic(base/'request.json',request)
    assert request['independent_confirmation']
    old=read(SCREEN/'request.json')['bank'];bank=request['bank']
    assert bank['noise_sha256']!=old['noise_sha256'] and bank['label_sha256']!=old['label_sha256']
    assert not np.array_equal(np.load(base/'inputs/noise.npy',mmap_mode='r')[:8],
        np.load(SCREEN/'inputs/noise.npy',mmap_mode='r')[:8])


def pipeline():
    configure();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'STOP_AFTER_CURRENT').exists():return
        prepare();engine.run_stage(STAGE)
        base=ROOT/STAGE
        if read(base/'status.json')['phase']!='complete':return
        rows=read(base/'results.json')
        analysis.audit_stage(base,[r['arm'] for r in rows if r['complete']])
        write_progress(base,read(base/'request.json'),rows)
        atomic(ROOT/'status.json',dict(phase='complete',total_arms=len(rows),
            numerical_failures=sum(not r['complete'] for r in rows),final_audit_passed=True))


if __name__=='__main__':
    configure();p=argparse.ArgumentParser();a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--pipeline',action='store_true');a.add_argument('--worker',type=int)
    p.add_argument('--stage');p.add_argument('--run-id');p.add_argument('--parent-pid',type=int)
    args=p.parse_args()
    if args.pipeline:pipeline()
    else:engine.worker(args.stage,args.worker,args.run_id,args.parent_pid)
