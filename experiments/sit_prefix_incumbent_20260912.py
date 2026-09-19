"""Replay one previously confirmed IG incumbent on the new 5K confirmation bank."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from experiments.sit_prefix_coarse_20260912 import pipeline as prior
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import WORK,EXPS,atomic,read,sha

ROOT=EXPS/'sit_prefix_incumbent_20260912'
CONFIRM=EXPS/'sit_prefix_confirmation_20260912/prefix_confirm_5k'
OLD=EXPS/'sit_guidance_fusion_20260910/selected_5k'
STAGE='prefix_incumbent_5k'
MODULE='experiments.sit_prefix_incumbent_20260912'
PROTOCOL=WORK/'docs/SIT_PREFIX_INCUMBENT_PROTOCOL_20260912_ZH.md'
REPORT=WORK/'docs/SIT_PREFIX_INCUMBENT_RESULTS_20260912_ZH.md'


def selected():
    old=read(OLD/'request.json')
    config=next(c for c in old['configs'] if c['arm']=='ig_local_residual_11')
    return dict(config,parameters=dict(config['parameters'],inherited_exact=True))


def sources():return sorted(set([*prior.sources(),Path(__file__).resolve(),PROTOCOL]))


def verify_parent():
    parent=prior.verify_parent();assets=dict(parent['assets'])
    for base in (OLD,CONFIRM):
        request=read(base/'request.json');assert read(base/'analysis_audit.json')['passed']
        for category in ('sources','assets','references'):
            for p,digest in request[category].items():assert sha(p)==digest,p
        for name,digest in request['bank_files'].items():assert sha(base/'inputs'/name)==digest,name
        for name in ('request.json','results.json','analysis_audit.json'):assets[str(base/name)]=sha(base/name)
    rows={r['arm']:r for r in read(CONFIRM/'results.json')}
    candidate=rows['ig_prefix_native_00']
    assert all(r['complete'] for r in rows.values())
    assert all(candidate['fid']<rows[name]['fid'] for name in ('ig_prefix_original_00','ig_prefix_local_00'))
    return dict(parent,assets=assets)


def configure():
    engine.ROOT,engine.MODULE,engine.PROTOCOL=ROOT,MODULE,PROTOCOL
    engine.STAGES={STAGE:(5000,2026091205)}
    engine.fusion=SimpleNamespace(FAMILIES=prior.core.previous.FAMILIES,sample=prior.core.sample,
        install=prior.core.install,configurations=prior.core.configurations,limiting_checks=prior.core.limiting_checks)
    engine.additional_sources,engine.verify_parent=sources,verify_parent
    analysis.write_progress=write_progress


def write_progress(base,request,rows):
    lines=['# 固定历史融合IG：同一5K补充对照\n',f'已完成{len(rows)}/1组。与native弱前缀确认共用同一5K，不是另一个噪声复验。\n',
        '|方法|FID|sFID|IS|Full/prefix|采样及解码GPU秒|','|---|--:|--:|--:|--:|--:|']
    for r in rows:
        if not r['complete']:lines.append('|历史融合IG|数值失败|—|—|—|—|');continue
        m=r['metrics']
        lines.append(f'|历史融合IG|{r["fid"]:.6f}|{m["sfid"]:.6f}|{m["inception_score"]:.4f}|'
            f'{r["full_calls_per_image"]:.0f}/{r["prefix_calls_per_image"]:.0f}|{r["sum_batch_gpu_seconds"]:.2f}|')
    lines+=['\n[固定补充协议](SIT_PREFIX_INCUMBENT_PROTOCOL_20260912_ZH.md) · '
        '[三组确认结果](SIT_PREFIX_CONFIRMATION_RESULTS_20260912_ZH.md)\n',
        f'请求SHA256：{sha(base/"request.json")}。\n']
    output='\n'.join(lines);(base/'report.md').write_text(output);REPORT.write_text(output)


@torch.inference_mode()
def preflight():
    verify_parent();prior.core.install();rt=engine.parent.operators.make_runtime()
    noise=torch.from_numpy(np.load(OLD/'inputs/noise.npy')[:8].copy()).cuda()
    labels=torch.from_numpy(np.load(OLD/'inputs/labels.npy')[:8].copy()).cuda()
    config=selected();z,stats=prior.core.sample(rt,noise,labels,config)
    path=OLD/config['arm']/'rank0/batch0000.npz'
    with np.load(path) as saved:np.testing.assert_array_equal(z.cpu().numpy(),saved['latents'])
    assert stats['full_calls']==128 and stats['prefix_calls']==64
    atomic(ROOT/'development_check.json',dict(passed=True,old_trajectory_exact=True,
        golden_path=str(path),golden_sha256=sha(path),source_hashes={str(p):sha(p) for p in sources()},
        no_fid_used=True))


def pipeline():
    configure();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'STOP_AFTER_CURRENT').exists():return
        verify_parent()
        if not (ROOT/'development_check.json').exists():preflight()
        base=ROOT/STAGE;existed=(base/'request.json').exists()
        selection=dict(reason='one stronger previously confirmed incumbent; fixed before current candidate result',
            matched_confirmation_request_sha256=sha(CONFIRM/'request.json'),
            old_request_sha256=sha(OLD/'request.json'),independent_noise_replications=1)
        engine.prepare_stage(STAGE,[selected()],selection=selection)
        request=read(base/'request.json')
        if not existed:
            request['independent_confirmation']=True
            request['same_bank_incumbent_supplement']=True
            atomic(base/'request.json',request)
        bank=request['bank'];other=read(CONFIRM/'request.json')['bank']
        assert bank['noise_sha256']==other['noise_sha256'] and bank['label_sha256']==other['label_sha256']
        for name in ('noise.npy','labels.npy'):assert sha(base/'inputs'/name)==sha(CONFIRM/'inputs'/name)
        del request
        engine.run_stage(STAGE)
        if read(base/'status.json')['phase']!='complete':return
        rows=read(base/'results.json');analysis.audit_stage(base,[r['arm'] for r in rows if r['complete']])
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
