"""Audit matched guidance curves, their costs, and selected fresh-noise comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments import small_sit_guidance_tuning_20260910 as study
from experiments import analyze_small_sit_carrier_flow_20260909 as pictures
from experiments.analyze_official_sit_pfr_symmetric_20260909 import fid_components
from experiments.lifting_scale_sweep_20260909 import array_sha,atomic,read,sha

OUT=study.WORK/'docs/data/small_sit_guidance_tuning_20260910'
REPORT=study.WORK/'docs/SMALL_SIT_GUIDANCE_TUNING_RESULTS_20260910_ZH.md'
TITLES=dict(ig='IG',residual_norm='Residual direction',adg='ADG')


def independent_metrics(directory,references):
    values={}
    with np.load(directory/'activations.npz') as features:
        for key,(mean,covariance) in references.items():
            assert features[key].shape[0]==study.SAMPLES
            if study.SAMPLES<=2000:
                a,b=fid_components(features[key].astype(float),mean,covariance)
                values[key]=a+b
            else:
                from scipy.linalg import eigh
                from experiments.analyze_small_sit_predictable_gap_confirmation_20260909 import covariance_fid
                val,vec=eigh((covariance+covariance.T)*.5)
                assert val.min()>-1e-7
                root=(vec*np.sqrt(np.maximum(val,0)))@vec.T
                values[key]=covariance_fid(features[key],mean,covariance,root)
    return values


def main(require_complete):
    study.install_infrastructure()
    request,request_hash=study.verify_request()
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb')==request['inception_graph_sha256']
    status=read(study.ROOT/'status.json')
    records=read(study.ROOT/'results.json') if (study.ROOT/'results.json').exists() else []
    if require_complete:
        assert status['phase']=='complete' and len(records)==len(study.ARMS)
    OUT.mkdir(parents=True,exist_ok=True)
    noise=np.load(study.BANK_ROOT/'noise.npy',mmap_mode='r')
    labels=np.load(study.BANK_ROOT/'labels.npy')
    assert array_sha(noise)==request['bank']['noise_sha256']
    assert array_sha(labels)==request['bank']['label_sha256']
    with np.load(study.infrastructure.REFERENCE) as ref:
        references={name:(ref[m].astype(float),ref[c].astype(float))
            for name,m,c in (('pool_3','mu','sigma'),('spatial','mu_s','sigma_s'))}
    rows,checks=[],[]
    for record in records:
        if not record['complete']:
            continue
        arm=record['arm'];directory=study.ROOT/arm
        assert record['request_sha256']==request_hash and arm in study.ARMS
        assert record['coverage_verified'] and record['samples']==study.SAMPLES
        assert record['noise_sha256']==request['bank']['noise_sha256']
        assert record['label_sha256']==request['bank']['label_sha256']
        for name,key in (('samples.npz','sample_sha256'),('latents.npy','latents_sha256'),
                         ('activations.npz','activations_sha256')):
            assert sha(directory/name)==record[key]
        seen,calls,geometry_calls,seconds=set(),0,0,0.
        diagnostics=np.zeros(len(study.DIAGNOSTICS))
        geometry_by_block=np.zeros(len(study.ALPHAS))
        diagnostics_by_block=np.zeros((len(study.ALPHAS),len(study.DIAGNOSTICS)))
        for rank in range(study.RANKS):
            d=directory/f'rank{rank}';summary=read(d/'summary.json')
            assert summary['complete'] and summary['request_sha256']==request_hash
            assert summary['noise_sha256']==request['bank']['noise_sha256']
            assert summary['label_sha256']==request['bank']['label_sha256']
            for entry in summary['files']:
                start,path=entry['start'],d/entry['file']
                assert start not in seen and (start//study.BATCH)%study.RANKS==rank
                seen.add(start);assert sha(path)==entry['sha256']
                with np.load(path) as batch:
                    assert str(batch['request_sha256'])==request_hash
                    assert str(batch['noise_sha256'])==array_sha(noise[start:start+study.BATCH])
                    np.testing.assert_array_equal(batch['labels'],labels[start:start+study.BATCH])
                    assert np.isfinite(batch['latents']).all()
                    assert int(batch['prefix_calls'])==int(batch['auxiliary_full_calls'])==0
                    assert int(batch['full_calls'])==int(batch['block_full_calls'].sum())
                    calls+=int(batch['full_calls'])
                    seconds+=float(batch['trajectory_seconds'])+float(batch['decode_seconds'])
                    count=batch['block_geometry_calls']
                    assert int(count.sum())==int(batch['geometry_calls'])
                    if study.SPECS[arm]['family']=='ig':assert not count.any()
                    else:
                        np.testing.assert_array_equal(count[:4],batch['block_full_calls'][:4])
                        assert count[4]==0
                    geometry_calls+=int(count.sum())
                    weighted=batch['block_geometry_diagnostics']*count[:,None,None]
                    assert np.isfinite(weighted).all()
                    diagnostics+=weighted.sum(axis=(0,1))
                    diagnostics_by_block+=weighted.sum(axis=1)
                    geometry_by_block+=count*study.BATCH
        assert seen==set(range(0,study.SAMPLES,study.BATCH))
        assert abs(seconds-record['sum_batch_gpu_seconds'])<1e-7
        assert calls/(study.SAMPLES/study.BATCH)==record['full_calls_per_image']
        key=dict(activations_sha256=record['activations_sha256'],reference_sha256=request['reference_sha256'],
                 fid=record['fid'],sfid=record['metrics']['sfid'])
        cache=OUT/f'audit_{arm}.json'
        if cache.exists():
            check=read(cache);assert check['key']==key
        else:
            values=independent_metrics(directory,references)
            check=dict(key=key,independent_fid=values['pool_3'],independent_sfid=values['spatial'],
                fid_absolute_error=abs(values['pool_3']-record['fid']),
                sfid_absolute_error=abs(values['spatial']-record['metrics']['sfid']))
            assert check['fid_absolute_error']<.001 and check['sfid_absolute_error']<.001
            atomic(cache,check)
        checks.append(dict(arm=arm,coverage_passed=True,**check))
        diag=diagnostics/(geometry_calls*study.BATCH) if geometry_calls else np.zeros(len(study.DIAGNOSTICS))
        rows.append(dict(arm=arm,**study.SPECS[arm],fid=record['fid'],sfid=record['metrics']['sfid'],
            inception_score=record['metrics']['inception_score'],full_calls_per_image=record['full_calls_per_image'],
            geometry_calls_per_image=geometry_calls/(study.SAMPLES/study.BATCH),sum_batch_gpu_seconds=seconds,
            **dict(zip(study.DIAGNOSTICS,map(float,diag)))))
        atomic(OUT/f'geometry_{arm}.json',dict(diagnostic_names=study.DIAGNOSTICS,
            query_weighted_mean=dict(zip(study.DIAGNOSTICS,map(float,diag))),
            per_block=(diagnostics_by_block/np.maximum(geometry_by_block[:,None],1)).tolist(),
            block_query_image_counts=geometry_by_block.tolist(),includes_rejected_ode_queries=True))
    preflights=[read(p) for p in sorted(study.ROOT.glob('preflight_rank*.json'))]
    assert all(p['passed'] and p['pair_prefix_exact'] and p['zero_three_exact'] for p in preflights)
    assert all(all(g['exact'] for g in p['golden']) for p in preflights)
    assert all(all(g['exact'] for g in p.get('selected_old_endpoints_exact',[])) for p in preflights)
    if require_complete:assert len(preflights)==study.RANKS
    best={f:min((r for r in rows if r['family']==f),key=lambda r:r['fid'])
          for f in study.FAMILIES if any(r['family']==f for r in rows)}
    parity=[]
    if study.SAMPLES==1000:
        for arm,directory in (('ig_a070',study.residual.OLD_ROOT/'ig_restarted'),
                              ('residual_norm_a070',study.residual.ROOT/'residual_norm')):
            selected=[r for r in records if r['arm']==arm and r['complete']]
            if not selected:continue
            r=selected[0];old=read(directory/'result.json')
            assert sha(directory/'latents.npy')==old['latents_sha256']
            assert sha(directory/'samples.npz')==old['sample_sha256']
            np.testing.assert_array_equal(np.load(directory/'latents.npy'),np.load(study.ROOT/arm/'latents.npy'))
            with np.load(directory/'samples.npz') as a,np.load(r['sample_path']) as b:
                np.testing.assert_array_equal(a['arr_0'],b['arr_0'])
            parity.append(dict(arm=arm,all_latents_exact=True,all_pixels_exact=True,
                images=study.SAMPLES,previous_fid=old['fid'],current_fid=r['fid'],
                fid_repeat_difference=r['fid']-old['fid']))
    pictures.OUT,pictures.BANK_ROOT=OUT,study.BANK_ROOT
    selected_arms={r['arm'] for r in best.values()}
    examples=pictures.contact_sheet([r for r in records if r['arm'] in selected_arms]) if rows else None
    comparisons=[]
    if 'ig' in best:
        baseline=best['ig']
        for family,r in best.items():
            comparisons.append(dict(family=family,arm=r['arm'],baseline=baseline['arm'],
                fid_difference=r['fid']-baseline['fid'],
                fid_reduction_percent=100*(1-r['fid']/baseline['fid']),
                cost_change_percent=100*(r['sum_batch_gpu_seconds']/baseline['sum_batch_gpu_seconds']-1)))
    audit=dict(status=status,request_sha256=request_hash,rows=rows,arithmetic_checks=checks,
        preflights=preflights,cpu_checks=request['cpu_checks'],old_anchor_parity=parity,
        best_within_tested_grid=best,best_comparisons=comparisons,
        fixed_examples=examples,source_asset_bank_hashes_passed=True,
        independent_confirmation=request.get('independent_confirmation',False),
        independent_feature_extraction=False,research_goal_achieved=False)
    atomic(OUT/'audit.json',audit)
    if rows:
        with (OUT/'metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 小SiT：调过强度后的IG、残差方向与ADG','',
        f"状态：`{status['phase']}`，已完成 {len(rows)}/{len(study.ARMS)} 组，每组 {study.SAMPLES} 张。",
        '完整表保留每个预先指定的配置；FID越低越好。', '']
    if best:
        winner=min(best.values(),key=lambda r:r['fid'])
        lines.extend([f"本轮最低FID为 `{winner['arm']}`：**{winner['fid']:.6f}**。",''])
        if 'ig' in best:
            baseline=best['ig']
            lines.append(f"IG当前已测最低为 `{baseline['arm']}`：{baseline['fid']:.6f}。")
        if study.SAMPLES==1000:
            lines.extend(['这是重复使用探索噪声后的参数选择，不能算独立确认；仅比较已测的固定时间轮廓和强度网格。',
                '若最优位于边界，仍需检查范围；候选超过IG时再冻结两者，以新噪声5K验证。',''])
        else:
            lines.extend(['本轮参数由旧1K选择后冻结，新噪声与平衡类别均在确认前生成；模型、训练数据与真实图参考未独立更换。',''])
    lines.extend(['## 各强度FID','', '| 峰值额外强度 | IG | 残差方向 | ADG |',
        '|---:|---:|---:|---:|'])
    for peak in sorted({r['peak'] for r in rows}):
        values=[]
        for family in ('ig','residual_norm','adg'):
            match=[r for r in rows if r['family']==family and r['peak']==peak]
            values.append(f"{match[0]['fid']:.6f}" if match else '—')
        lines.append(f"| {peak:g} | "+' | '.join(values)+' |')
    lines.extend(['','## 完整指标与实测成本','',
        '| 配置 | FID | sFID | IS | 主干NFE/图 | 采样＋解码GPU秒 |',
        '|---|---:|---:|---:|---:|---:|'])
    for r in rows:
        lines.append(f"| {r['arm']} | {r['fid']:.6f} | {r['sfid']:.6f} | {r['inception_score']:.4f} | {r['full_calls_per_image']:.3f} | {r['sum_batch_gpu_seconds']:.3f} |")
    failures=[r for r in records if not r['complete']]
    if failures:lines.extend(['',f'数值失败（保留原配置与记录）：{failures}'])
    lines.extend(['','成本为各batch实测GPU采样＋解码秒之和，包含诊断、新投影或角度算术以及自适应步数变化；不含加载、预检、ADM评价和历史训练/拟合。',
        '每次RHS仅一次主干双头前向，额外主干调用为0。残差方法使用此前冻结的6160系数线性读出；不将其称为非线性预测组合。',
        'ADG为已有论文公式在当前Strong/Weak上的移植；它是本轮非线性比较，不能视作新方法。','',
        '## 完整性核验','',
        '核对冻结源码、权重、参考、输入、每批输出、汇总样本与Inception特征哈希；检查所有图索引无重复无遗漏。',
        '四rank预检验证Strong/Weak入口一致、零强度恢复以及旧IG/残差工作点逐位复现。',
        'FID/sFID用已保存的Inception特征独立以FP64重算；它不等于独立提取特征。'])
    if checks:
        lines.append(f"最大绝对复算差：FID {max(x['fid_absolute_error'] for x in checks):.3g}；sFID {max(x['sfid_absolute_error'] for x in checks):.3g}。")
    for p in parity:
        lines.append(f"`{p['arm']}`全部{study.SAMPLES}张latent和像素与历史逐位相同，重复评价FID差 {p['fid_repeat_difference']:+.6f}；千分级变化不当成质量收益。")
    lines.extend(['','## 证据','',f'- [冻结请求]({study.ROOT}/request.json)',
        f'- [原始指标]({study.ROOT}/results.json)',f'- [审计JSON]({OUT}/audit.json)',
        f'- [完整CSV]({OUT}/metrics.csv)',f'- [采样代码]({study.WORK}/experiments/{study.__name__.split(".")[-1]}.py)',
        f'- [分析代码]({Path(__file__).resolve()})'])
    if examples:
        lines.extend(['', '三个族各自最优配置的相同前8个输入，未按图片质量挑选：','',f"![固定样例]({examples['path']})"])
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(REPORT),audit=str(OUT/'audit.json'),rows=len(rows),
        best=best,comparisons=comparisons),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();main(args.require_complete)
