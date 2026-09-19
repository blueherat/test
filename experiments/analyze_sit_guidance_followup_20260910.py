"""Source-backed summaries and independent cached-feature audits for follow-up stages."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from experiments.lifting_scale_sweep_20260909 import EXPS,WORK,array_sha,atomic,read,sha
from experiments.analyze_official_sit_pfr_symmetric_20260909 import fid_components
from experiments.analyze_small_sit_predictable_gap_confirmation_20260909 import covariance_fid

REFERENCE_CACHE={}
REPORTS={
    'tuning_1k':'SIT_GUIDANCE_FUSION_TUNING_RESULTS_20260910_ZH.md',
    'selected_5k':'SIT_GUIDANCE_FUSION_CONFIRMATION_RESULTS_20260910_ZH.md',
}


def references(request):
    h=request['reference_sha256']
    if h not in REFERENCE_CACHE:
        output={}
        with np.load(request['reference']) as data:
            for key,mu,cov in [('pool_3','mu','sigma'),('spatial','mu_s','sigma_s')]:
                mean,matrix=data[mu].astype(float),data[cov].astype(float)
                val,vec=eigh((matrix+matrix.T)*.5);assert val.min()>-1e-7
                root=(vec*np.sqrt(np.maximum(val,0)))@vec.T
                output[key]=(mean,matrix,root)
        REFERENCE_CACHE[h]=output
    return REFERENCE_CACHE[h]


def audit_stage(base,selected_arms):
    base=Path(base);request=read(base/'request.json');h=sha(base/'request.json')
    rows=read(base/'results.json');n=request['samples'];configs={c['arm']:c for c in request['configs']}
    assert read(base/'status.json')['phase']=='complete' and len(rows)==len(configs)
    assert {r['arm'] for r in rows}==set(configs)
    for category in ('sources','assets','references','preflight_reference_files'):
        for path,digest in request.get(category,{}).items():assert sha(path)==digest,path
    bank=Path(request['bank_root'])
    for name,digest in request['bank_files'].items():assert sha(bank/name)==digest,name
    assert sha(request['reference'])==request['reference_sha256']
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb')==request['inception_graph_sha256']
    noise=np.load(bank/'noise.npy',mmap_mode='r');labels=np.load(bank/'labels.npy')
    assert array_sha(noise)==request['bank']['noise_sha256'] and array_sha(labels)==request['bank']['label_sha256']
    out=WORK/'docs/data'/('sit_guidance_fusion_'+base.name if base.name in REPORTS else base.name)
    out.mkdir(parents=True,exist_ok=True);selected=set(selected_arms);audits=[];manifests=0
    for row in rows:
        directory=base/row['arm'];config=configs[row['arm']]
        assert all(row[k]==v for k,v in config.items())
        assert row['request_sha256']==h and read(directory/'result.json')==row
        if not row['complete']:
            assert row['fid'] is None and (directory/'numerical_failure.json').exists();continue
        assert row['samples']==n and row['coverage_verified']
        assert row['noise_sha256']==request['bank']['noise_sha256'] and row['label_sha256']==request['bank']['label_sha256']
        assert row['metrics']==read(directory/'fid.json') and row['metrics']['sample_count']==n
        assert row['fid']==row['metrics']['fid'] and np.isfinite(row['fid'])
        detailed=row['arm'] in selected
        if detailed:
            for name,key in [('samples.npz','sample_sha256'),('latents.npy','latents_sha256'),('activations.npz','activations_sha256')]:
                assert sha(directory/name)==row[key],directory/name
        seen=set();seconds=0.;calls=np.zeros(3)
        for rank in range(4):
            shard=directory/f'rank{rank}';summary=read(shard/'summary.json')
            assert summary['complete'] and summary['request_sha256']==h
            assert summary['noise_sha256']==row['noise_sha256'] and summary['label_sha256']==row['label_sha256']
            for rec in summary['files']:
                start=rec['start'];assert start not in seen and (start//8)%4==rank
                seen.add(start)
                if not detailed:continue
                path=shard/rec['file'];assert sha(path)==rec['sha256'],path
                with np.load(path) as batch:
                    assert str(batch['request_sha256'])==h
                    assert str(batch['noise_sha256'])==array_sha(noise[start:start+8])
                    np.testing.assert_array_equal(batch['labels'],labels[start:start+8])
                    assert batch['latents'].shape==(8,4,32,32) and np.isfinite(batch['latents']).all()
                    assert batch['arr_0'].shape==(8,256,256,3) and batch['arr_0'].dtype==np.uint8
                    seconds+=float(batch['trajectory_seconds'])+float(batch['decode_seconds'])
                    calls+=np.array([int(batch[k]) for k in ('full_calls','prefix_calls','auxiliary_full_calls')])
        assert seen==set(range(0,n,8));manifests+=len(seen)
        if not detailed:continue
        assert abs(seconds-row['sum_batch_gpu_seconds'])<1e-7
        for value,key in zip(calls/(n/8),('full_calls_per_image','prefix_calls_per_image','auxiliary_full_calls_per_image')):
            assert value==row[key]
        cache=out/f'audit_{row["arm"]}.json'
        identity=dict(activations_sha256=row['activations_sha256'],reference_sha256=request['reference_sha256'],
            fid=row['fid'],sfid=row['metrics']['sfid'],samples=n)
        if cache.exists():
            check=read(cache);assert check['identity']==identity
        else:
            values={}
            with np.load(directory/'activations.npz') as data:
                for key,(mu,cov,root) in references(request).items():
                    assert data[key].shape[0]==n
                    values[key]=(sum(fid_components(data[key].astype(float),mu,cov)) if n<=2000
                                 else covariance_fid(data[key],mu,cov,root))
            check=dict(identity=identity,independent_fid=values['pool_3'],independent_sfid=values['spatial'],
                fid_absolute_error=abs(values['pool_3']-row['fid']),sfid_absolute_error=abs(values['spatial']-row['metrics']['sfid']))
            assert check['fid_absolute_error']<.001 and check['sfid_absolute_error']<.001,check
            atomic(cache,check)
        audits.append(dict(arm=row['arm'],raw_batch_and_aggregate_hashes_verified=True,costs_reconciled=True,**check))
        print(json.dumps(dict(audited=base.name,arm=row['arm'],fid=row['fid'])),flush=True)
    assert {a['arm'] for a in audits}=={r['arm'] for r in rows if r['complete'] and r['arm'] in selected}
    result=dict(passed=True,request_sha256=h,results=len(rows),samples_per_config=n,
        metadata_and_coverage_all_complete_arms=True,covered_batch_indices=manifests,
        raw_and_metric_audits=audits,feature_extraction_repeated=False,
        raw_bytes_rehashed_for_selected_only=True,independent_confirmation=request.get('independent_confirmation',False))
    atomic(out/'audit.json',result);atomic(base/'analysis_audit.json',result)
    if base.name in REPORTS:write_progress(base,request,rows)
    return result


def write_progress(base,request,rows):
    base=Path(base);valid=[r for r in rows if r['complete']]
    fields=['arm','family','role','source','key','strength','theta','parameters','fid','sfid','inception_score',
        'full_calls_per_image','prefix_calls_per_image','auxiliary_full_calls_per_image','sum_batch_gpu_seconds','complete']
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=fields,extrasaction='ignore');writer.writeheader()
    for row in rows:
        value=dict(row)
        if row['complete']:value.update({k:row['metrics'][k] for k in ('sfid','inception_score')})
        value['parameters']=json.dumps(row.get('parameters',{}),sort_keys=True);writer.writerow(value)
    (base/'results.csv').write_text(output.getvalue())
    def best(predicate):
        pool=[r for r in valid if predicate(r)]
        return min(pool,key=lambda r:r['fid']) if pool else None
    native={s:best(lambda r:r['source']==s and r['role']=='native' and r['family']!='strong') for s in ('ig','cfg')}
    component={s:best(lambda r:r['source']==s and r['role']!='fusion' and r['family']!='strong') for s in ('ig','cfg')}
    lines=[f'# Guidance 融合：{request["samples"]} 图阶段\n',
        f'已提交 {len(rows)}/{len(request["configs"])} 组，数值失败 {sum(not r["complete"] for r in rows)}。原始输出：`{base}`。\n',
        '本表 Δ 为候选减基线，负值更好。原生基线与单组件基线都取本轮各自已完成曲线最低 FID。耗时为采样＋解码累计 batch GPU 秒；不同方法的实际 Full/prefix 另列。\n']
    for source in ('ig','cfg'):
        if native[source]:lines.append(f'{source.upper()} 原生最佳：{native[source]["arm"]}，FID={native[source]["fid"]:.6f}。\n')
        if component[source]:lines.append(f'{source.upper()} 非融合组件最佳：{component[source]["arm"]}，FID={component[source]["fid"]:.6f}。\n')
    lines.extend(['|Family|已跑/计划|最低FID|Δ原生|Δ最佳组件|strength / theta|Full/prefix|采样+解码秒|',
        '|---|--:|--:|--:|--:|---|---|--:|'])
    for family in dict.fromkeys(c['family'] for c in request['configs']):
        row=best(lambda r:r['family']==family)
        if row is None:continue
        source=row['source'];delta=lambda b:f'{row["fid"]-b["fid"]:+.5f}' if b else '—'
        done=sum(r['family']==family for r in rows);planned=sum(c['family']==family for c in request['configs'])
        lines.append(f'|{family}|{done}/{planned}|{row["fid"]:.6f}|{delta(native[source])}|{delta(component[source])}|{row["strength"]} / {row["theta"]}|{row["full_calls_per_image"]:.2f}/{row["prefix_calls_per_image"]:.2f}|{row["sum_batch_gpu_seconds"]:.3f}|')
    if request['samples']==1000:
        lines.append('\n这是新 paired 1K 调参结果，不是独立确认。只有低于最佳非融合组件的融合方案才进入另一套新 5K；未赢过单组件时保留负结果并停止这一轮。网格边界胜者不称全局最优。\n')
    else:
        lines.append('\n这是从另一套 1K bank 固定配置后，在全新 5K 噪声上的确认。训练数据和 ADM 参考未更换；不将新噪声复验称作跨数据集验证。\n')
        lines.extend(['|配置|FID|sFID|IS|','|---|--:|--:|--:|'])
        for row in valid:lines.append(f'|{row["arm"]}|{row["fid"]:.6f}|{row["metrics"]["sfid"]:.6f}|{row["metrics"]["inception_score"]:.6f}|')
    audit=base/'analysis_audit.json'
    if audit.exists():
        info=read(audit)
        lines.append(f'\n完整元数据与覆盖核验通过；另对 {len(info["raw_and_metric_audits"])} 个配置逐批复核原始/聚合文件哈希、成本及缓存特征独立 FP64 FID/sFID。没有重新提取 Inception 特征。\n')
    lines.append(f'\n请求 SHA256：`{sha(base/"request.json")}`。完整逐组数据：`{base/"results.csv"}`。\n')
    text='\n'.join(lines);(base/'report.md').write_text(text)
    if base.name in REPORTS:(WORK/'docs'/REPORTS[base.name]).write_text(text)


def initial_confirmation_report():
    base=EXPS/'sit_guidance_portfolio_confirmation_20260910'
    request=read(base/'request.json');rows=read(base/'results.json')
    audit=audit_stage(base,[r['arm'] for r in rows])
    by_arm={r['arm']:r for r in rows}
    heun=by_arm['control_ig_heun64_05'];dopri=by_arm['control_ig_dopri5_05']
    local=by_arm['i40_ig_local_attention_g2_p1'];cfg=by_arm['control_cfg_c75_04']
    gain=lambda r,b:(1-r['fid']/b['fid'])*100
    cost=lambda r,b:r['sum_batch_gpu_seconds']/b['sum_batch_gpu_seconds']
    lines=['# Portfolio 固定候选：新 5K 确认\n',
        f'七组均完成，每组新噪声 5000 图。#40 FID **{local["fid"]:.6f}**，原生 Heun IG **{heun["fid"]:.6f}**，改善 **{gain(local,heun):.3f}%**，采样/解码成本 **{cost(local,heun):.3f}×**。Dopri5 IG 为 **{dopri["fid"]:.6f}**，#40 相对它改善 **{gain(local,dopri):.3f}%**。\n',
        f'原生 CFG FID **{cfg["fid"]:.6f}**。以下各候选参数完全固定自之前 1K，未用本 5K 重新选择强度。\n',
        '|配置|FID↓|sFID↓|IS↑|相对同源基线FID改善|成本倍数|Full/prefix|',
        '|---|---:|---:|---:|---:|---:|---|']
    titles=['Heun IG','#40 局部注意力','Dopri5 IG','原生 CFG','#1 CFG+APG','#38 CFG 条件割线','#3 CFG 方差回缩']
    for title,row in zip(titles,rows):
        baseline=cfg if row['source']=='cfg' else heun
        lines.append(f'|{title}|{row["fid"]:.6f}|{row["metrics"]["sfid"]:.6f}|{row["metrics"]["inception_score"]:.6f}|{gain(row,baseline):+.3f}%|{cost(row,baseline):.3f}×|{row["full_calls_per_image"]:.3f}/{row["prefix_calls_per_image"]:.0f}|')
    lines.extend(['\n这里确认固定配置在新噪声 bank 上的表现；并未证明同成本最优、统计显著性、机制因果或新颖性。#1是已有APG适配，#40受空间上下文弱化的既有方法启发。后续融合使用另一套新1K，胜者再用第三套新5K。\n',
        '全部七组原始batch、聚合像素/latent/feature哈希、输入身份、覆盖和实际调用/计时核对通过；缓存特征独立FP64 FID/sFID重算误差均小于.001。没有重新抽取Inception特征。四卡采样前逐元素复现了七个旧配置的首批latent。\n',
        f'新噪声 seed=202610060，标签 seed=202610061，每类50张。noise SHA256=`{request["bank"]["noise_sha256"]}`。request SHA256=`{sha(base/"request.json")}`。\n',
        '[冻结协议](SIT_GUIDANCE_PORTFOLIO_CONFIRMATION_PROTOCOL_20260910_ZH.md) · [审计](data/sit_guidance_portfolio_confirmation_20260910/audit.json)。\n'])
    report=WORK/'docs/SIT_GUIDANCE_PORTFOLIO_CONFIRMATION_RESULTS_20260910_ZH.md'
    report.write_text('\n'.join(lines));print(json.dumps(dict(initial_confirmation_report=str(report))),flush=True)
    return audit


if __name__=='__main__':initial_confirmation_report()
