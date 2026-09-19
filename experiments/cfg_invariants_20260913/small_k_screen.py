"""Three K=.03 follow-ups; immutable parent sampler and paired input bank."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from experiments.cfg_invariants_20260913 import image_screen as base

HERE=Path(__file__).resolve()
MAIN_ROOT=base.ROOT
ROOT=MAIN_ROOT/'small_k_0p03'
ARMS=('ctrl_author','ctrl_gap_project','ctrl_physical')


def configure():
    base.ROOT=ROOT
    base.K=.03
    base.ARMS=ARMS


def prepare():
    original=base.c.read(MAIN_ROOT/'request.json')
    for group in ('sources','assets','inputs'):
        for path,digest in original[group].items():assert base.c.sha(path)==digest,path
    configure();ROOT.mkdir(exist_ok=True)
    bank=ROOT/'screen_bank.npz'
    if not bank.exists():shutil.copyfile(MAIN_ROOT/'screen_bank.npz',bank)
    assert base.c.sha(bank)==base.c.sha(MAIN_ROOT/'screen_bank.npz')
    # Runtime requires a profile receipt; these three arms use no profiles.
    if not (ROOT/'profiles.npz').exists():base.atomic_npz(ROOT/'profiles.npz')
    request=dict(original)
    request.update(arms=list(ARMS),ctrl_K=.03,calibration_samples=0,
        parent_request=str(MAIN_ROOT/'request.json'),parent_request_sha256=base.c.sha(MAIN_ROOT/'request.json'),
        followup_reason='Public K=.3 is large in SiT velocity units; one prespecified 10x-smaller K sanity check, not a broad search',
        matched_controls='Main vanilla/half CFG reused only as external paired references; these three candidate images/FID/classification are newly computed',
        inputs={str(bank):base.c.sha(bank)},
        sources=dict(original['sources'],**{str(HERE):base.c.sha(HERE)}))
    path=ROOT/'request.json'
    if path.exists():assert base.c.read(path)==request
    else:base.c.atomic(path,request)
    base.c.atomic(ROOT/'calibration.json',dict(complete=True,generated_paths=0,
        sha256=base.c.sha(ROOT/'profiles.npz'),request_sha256=base.c.sha(path)))
    # The only changed numeric coefficient is K; verify against authors' helper.
    import importlib.util
    spec=importlib.util.spec_from_file_location('smallk_author_check',base.AUTHOR)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    rng=torch.Generator().manual_seed(base.SEED)
    g=torch.randn((3,4,8,8),generator=rng);old=torch.randn(g.shape,generator=rng)
    state=module.CFGCtrlState(prev_guidance_eps=old)
    params=module.CFGCtrlParams(smc_cfg_enable=True,smc_cfg_K=.03)
    expected=module.CFGCtrlMixin()._cfg_ctrl_apply(noise_pred_posi=g,noise_pred_nega=torch.zeros_like(g),
             cfg_scale=1.,progress_id=2,params=params,state=state)
    assert torch.equal(base.author_gap(g,old),expected)
    base.c.atomic(ROOT/'cpu_check.json',dict(author_rule_exact_at_K_0p03=True,cuda_used=False))
    print(json.dumps(dict(prepared=True,root=str(ROOT),K=base.K,arms=ARMS)),flush=True)


def report():
    configure();rows=[]
    for arm in ARMS:
        folder=ROOT/arm
        if not (folder/'summary.json').exists():continue
        row=base.c.read(folder/'summary.json')
        for name in ('fid','classifier'):
            if (folder/(name+'.json')).exists():row.update(base.c.read(folder/(name+'.json')))
        rows.append(row)
    baseline={}
    for name in ('summary','fid','classifier'):
        path=MAIN_ROOT/'cfg_base'/(name+'.json')
        if path.exists():baseline.update(base.c.read(path))
    comparisons=[]
    if 'fid' in baseline:
        for row in rows:
            if 'fid' in row:
                comparisons.append(dict(arm=row['arm'],K=.03,fid=row['fid'],baseline_fid=baseline['fid'],
                                        delta_vs_cfg=row['fid']-baseline['fid']))
    lookup={row['arm']:row for row in rows}
    for candidate in ('ctrl_gap_project','ctrl_physical'):
        if candidate not in lookup or 'ctrl_author' not in lookup:continue
        candidate_row,author_row=lookup[candidate],lookup['ctrl_author']
        pair=dict(candidate=candidate,control='ctrl_author',K=.03)
        if 'fid' in candidate_row and 'fid' in author_row:
            pair['fid_delta_candidate_minus_author']=candidate_row['fid']-author_row['fid']
        paths=[ROOT/name/'classifier_probabilities.npz' for name in (candidate,'ctrl_author')]
        if all(path.exists() for path in paths):
            values=[dict(np.load(path)) for path in paths]
            diff=values[0]['target_probability']-values[1]['target_probability']
            rng=np.random.default_rng(base.SEED+2)
            boot=diff[rng.integers(0,base.N,(1000,base.N))].mean(1)
            pair.update(target_probability_paired_difference=float(diff.mean()),
                        target_probability_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
        comparisons.append(pair)
    base.c.atomic(ROOT/'comparisons.json',comparisons)
    output=io.StringIO();fields=['arm','fid','sfid','inception_score','target_top1','target_probability',
        'mean_parallel_extra','mean_extra_to_native_norm','mean_off_gap_fraction','seconds']
    writer=csv.DictWriter(output,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (ROOT/'results.csv').write_text(output.getvalue())
    examples=[];labels=[]
    for parent,arm,label in [(MAIN_ROOT,'cfg_base','cfg_base')]+[(ROOT,a,a+' K=.03') for a in ARMS]:
        path=parent/arm/'samples.npz'
        if path.exists():
            examples.extend(np.load(path)['arr_0'][:8]);labels.extend([label]*8)
    if examples:base.gallery(np.stack(examples),ROOT/'comparison_grid.png',cols=8,labels=labels)
    lines=['# K=.03 的定向补验','',
        '作者默认 K=.3 在 SiT velocity 单位上出现过强反转，因此仅补 K=.03 的三个配置，其他参数不变。',
        '这是确定量纲敏感性的补验，不是新一轮广泛调参。每个配置重新生成同一 bank 的400张，单输出224次前向；FID与分类器实算。',
        'vanilla基线来自主屏同噪声/类别，不重复计为新采样。小样本 FID 只筛选。',
        'ctrl_physical 使用原始 gap 历史与物理时间导数，而作者保存 modified gap；差异不能全部归因参数单位。',
        '历史在 Heun 两个 stage 内冻结，仅接受步后提交左端 proposal。单位变换的代数核查仅适用于可逆的中间时刻；clean 的 t=1 与 epsilon 的 t=0 是奇异端点。',
        '', '|arm|FID400↓|top1↑|mean extra∥|extra norm/native|off-gap|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in ([baseline] if baseline else [])+rows:
        vals=['—' if key not in row else f'{row[key]:.4f}' for key in ('fid','target_top1','mean_parallel_extra','mean_extra_to_native_norm','mean_off_gap_fraction')]
        lines.append('|'+row.get('arm','cfg_base')+'|'+'|'.join(vals)+'|')
    lines += ['',f'产物：[{ROOT.name}](<{ROOT}>)；[同seed图集](<{ROOT}/comparison_grid.png>)；[CSV](<{ROOT}/results.csv>)。','']
    (ROOT/'image_results.md').write_text('\n'.join(lines))
    (HERE.with_name('small_k_results.md')).write_text('\n'.join(lines))
    base.c.atomic(ROOT/'status.json',dict(complete=len(rows)==3 and all('fid' in row and 'target_top1' in row for row in rows),
               collected=len(rows),fid_evaluated=sum('fid' in row for row in rows),planned=3))
    print(json.dumps(dict(rows=len(rows),comparisons=comparisons)),flush=True)


def run():
    configure();base.verify()
    for phase in (base.worker,base.classify_outputs,base.evaluate):
        for rank in (0,1):phase(rank)
        report()
    report()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','report'])
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='run':run()
    else:report()
