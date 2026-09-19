"""Audit paired nonlinear-prediction samples and independently recompute FID."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments import small_sit_nonlinear_prediction_20260909 as study
from experiments import analyze_small_sit_carrier_flow_20260909 as pictures
from experiments.analyze_official_sit_pfr_symmetric_20260909 import fid_components
from experiments.lifting_scale_sweep_20260909 import array_sha,atomic,read,sha

OUT=study.WORK/'docs/data/small_sit_nonlinear_prediction_20260909'
REPORT=study.WORK/'docs/SMALL_SIT_NONLINEAR_PREDICTION_RESULTS_20260909_ZH.md'
TITLES=dict(ig_restarted='普通IG，本轮重跑',polar_exp='径向乘法＋有限旋转',
    sphere_exp='仅有限旋转',sphere_retraction='线性后等范数归一化')


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
                    if arm=='ig_restarted':assert not count.any()
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
        rows.append(dict(arm=arm,fid=record['fid'],sfid=record['metrics']['sfid'],
            inception_score=record['metrics']['inception_score'],full_calls_per_image=record['full_calls_per_image'],
            geometry_calls_per_image=geometry_calls/(study.SAMPLES/study.BATCH),sum_batch_gpu_seconds=seconds,
            **dict(zip(study.DIAGNOSTICS,map(float,diag)))))
        atomic(OUT/f'geometry_{arm}.json',dict(diagnostic_names=study.DIAGNOSTICS,
            query_weighted_mean=dict(zip(study.DIAGNOSTICS,map(float,diag))),
            per_block=(diagnostics_by_block/np.maximum(geometry_by_block[:,None],1)).tolist(),
            block_query_image_counts=geometry_by_block.tolist(),includes_rejected_ode_queries=True))
    preflights=[read(p) for p in sorted(study.ROOT.glob('preflight_rank*.json'))]
    assert all(p['passed'] and p['ordinary_ig_latents_exact'] and p['zero_all_four_exact'] for p in preflights)
    if require_complete:assert len(preflights)==study.RANKS
    old_parity=None
    if rows and 'old_baseline_for_parity' in request:
        old=request['old_baseline_for_parity'];current=records[0]
        assert old['arm']==current['arm']=='ig_restarted'
        assert old['request_sha256']==request['parent_request_sha256']
        assert sha(old['sample_path'])==old['sample_sha256']
        olddir=Path(old['sample_path']).parent
        assert sha(olddir/'latents.npy')==old['latents_sha256']
        np.testing.assert_array_equal(np.load(olddir/'latents.npy'),np.load(study.ROOT/'ig_restarted/latents.npy'))
        with np.load(old['sample_path']) as a,np.load(current['sample_path']) as b:
            np.testing.assert_array_equal(a['arr_0'],b['arr_0'])
        drift={}
        assert sha(olddir/'activations.npz')==old['activations_sha256']
        with np.load(olddir/'activations.npz') as a,np.load(study.ROOT/'ig_restarted/activations.npz') as b:
            for name in ('pool_3','spatial'):
                diff=a[name].astype(float)-b[name].astype(float)
                drift[name]=dict(exact=bool(np.array_equal(a[name],b[name])),
                    max_absolute=float(np.abs(diff).max()),rms=float(np.sqrt(np.square(diff).mean())))
        old_parity=dict(all_latents_exact=True,all_pixels_exact=True,images=study.SAMPLES,old_fid=old['fid'],
            recomputed_fid=current['fid'],fid_repeat_difference=current['fid']-old['fid'],feature_drift=drift)
    pictures.OUT,pictures.BANK_ROOT=OUT,study.BANK_ROOT
    examples=pictures.contact_sheet([r for r in records if r['complete']]) if rows else None
    audit=dict(status=status,request_sha256=request_hash,rows=rows,arithmetic_checks=checks,
        preflights=preflights,cpu_checks=request['cpu_checks'],old_baseline_parity=old_parity,
        fixed_examples=examples,source_asset_bank_hashes_passed=True,
        independent_confirmation=request.get('independent_confirmation',False),
        independent_feature_extraction=False,research_goal_achieved=False)
    atomic(OUT/'audit.json',audit)
    if rows:
        with (OUT/'metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fresh=request.get('independent_confirmation',False)
    lines=['# 小SiT：非线性预测组合的质量结果','',
        f"状态：{status['phase']}；完成{len(rows)}/{len(study.ARMS)}组，每组{study.SAMPLES}张。",'']
    if len(rows)==len(study.ARMS):
        base=rows[0];best=min(rows[1:],key=lambda r:r['fid'])
        if best['fid']<base['fid']:
            gain=100*(base['fid']-best['fid'])/base['fid']
            lines += [f"本轮最低候选为{TITLES[best['arm']]}，FID相对普通IG改善{gain:.3f}%。",
                '这是固定配置的新噪声确认结果；单模型单次5K仍不足以证明普遍收益。' if fresh else
                '这是复用探索bank上的选择结果，尚不能认定稳定收益；须另建新噪声确认。','']
        else:
            lines += ['全部非线性候选的FID都未改善普通IG。本轮固定构造的质量假说未获支持。',
                '不继续为这几项候选搜索角度、归一化强度或组合参数，也不把它扩大为所有非线性方法无效。','']
    lines += ['| 配置 | FID↓ | sFID↓ | IS↑ | Full/图 | 采样＋解码GPU秒 | 相对本轮IG成本 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {TITLES[r['arm']]} | {r['fid']:.6f} | {r['sfid']:.6f} | {r['inception_score']:.6f} | "
            f"{r['full_calls_per_image']:.3f} | {r['sum_batch_gpu_seconds']:.3f} | {r['sum_batch_gpu_seconds']/rows[0]['sum_batch_gpu_seconds']:.3f}x |")
    lines += ['', 'Strong/Weak、噪声和类别、引导强度、Dopri5容差、分段网格及解码器逐项匹配。',
        '每次RHS一次主干调用，零新增模型参数；自适应NFE及几何运算会影响实际成本。',
        '上表为跨batch累加的实测采样和解码GPU时间，包含逐查询诊断；不含加载、预检和FID评价，也不是四卡墙钟时间。','',
        '主候选在clean预测坐标中执行径向乘法与有限旋转，小强度一阶等于普通IG。',
        '纯旋转保持Strong预测长度；归一化对照先做线性IG再恢复Strong长度，因此后者本身包含线性外推。',
        '这几项不增加新模型信息，不证明潜变量posterior mean具有球面几何。','',
        '| 候选 | 平均转角参数/rad | 平均clean长度比 | 修正/原线性修正长度 | 偏离原仿射线的能量比例 | 退化回退比例 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows[1:]:
        lines.append(f"| {TITLES[r['arm']]} | {r['absolute_rotation_radians']:.6f} | {r['clean_norm_ratio']:.6f} | "
            f"{r['correction_to_linear_gap_norm']:.6f} | {r['off_affine_line_energy_fraction']:.6f} | {r['fallback_fraction']:.6f} |")
    lines += ['', '统计覆盖实际采样所有非零guidance的RHS查询（含Dopri5拒绝的查询），按查询和图像加权。',
        '仿射线残量针对 `v-S` 在 `S-W` 的正交分量。其非零说明该状态不能由一个标量IG精确表示，',
        '不等于证明整个输出分布不能由其他IG策略取得。完整分段诊断保存在geometry文件。','',
        '已核验完整batch覆盖、全部输入/源文件/权重/参考哈希；FP64独立复算FID和sFID。',
        '独立复算使用同一缓存Inception特征，不是独立特征提取或独立真实参考。',
        f"CPU稠密矩阵指数核对{request['cpu_checks']['matrix_exponential_cases']}例，最大clean误差{request['cpu_checks']['max_clean_error']:.3g}。"]
    if old_parity:
        lines += [f'本轮原生IG全部{study.SAMPLES}张latent及像素与历史基线逐位相同；成本为本轮重新测量。']
        lines += [f"同像素重复ADM评测的FID变化{old_parity['fid_repeat_difference']:+.6f}；两次缓存特征并非逐位相同，pool_3差异RMS为{old_parity['feature_drift']['pool_3']['rms']:.6g}。",
            '这项重复提取差异的具体根因尚未定位；独立FID复算核对的是本轮缓存特征。所有候选使用本轮IG作为对照，不把旧FID差异算作采样效果。']
    lines += ['', '用户指出构造缺乏生成直觉与理论依据后，本轮停止扩展。旋转/范数等式仅说明几何性质，不能决定为何应当采用该几何。',
        '后续[理论重审](IG_HYPOTHESIS_POSTERIOR_RETHINK_20260910_ZH.md)从近似推断偏差与候选后验出发，尚不是已验证的新方法。']
    lines += ['',f'[冻结协议]({study.PROTOCOL.name})；[完整数值](data/{OUT.name}/metrics.csv)；[核验记录](data/{OUT.name}/audit.json)。',
        '相关方法已有[ADG](https://arxiv.org/html/2506.11039v1)与[APG](https://arxiv.org/html/2410.02416v2)，本轮不作首次非线性guidance的主张。','',
        f'![固定最前8个相同输入](data/{OUT.name}/paired_first8.png)','',
        '固定最前8张用于核查实现与可见变化，不能代表整体质量。',
        f'原始文件：`{study.ROOT}`。原论文研究目标仍未完成。','']
    REPORT.write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(REPORT),completed=len(rows),rows=rows)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true')
    main(parser.parse_args().require_complete)
