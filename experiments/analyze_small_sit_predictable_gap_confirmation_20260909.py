"""Audit the fresh 5K residual-guidance comparison in feature covariance space."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from experiments import small_sit_predictable_gap_confirmation_20260909 as study
from experiments import analyze_small_sit_carrier_flow_20260909 as pictures
from experiments.lifting_scale_sweep_20260909 import array_sha, atomic, read, sha

OUT = study.WORK / 'docs/data/small_sit_predictable_gap_confirmation_20260909'
REPORT = study.WORK / 'docs/SMALL_SIT_PREDICTABLE_GAP_CONFIRMATION_RESULTS_20260909_ZH.md'


def covariance_fid(features, reference_mean, reference_covariance, reference_root):
    x = features.astype(np.float64)
    mean = x.mean(0)
    x -= mean
    covariance = x.T @ x / (len(x)-1)
    inner = reference_root @ covariance @ reference_root
    eigenvalues = eigh((inner+inner.T)*.5, eigvals_only=True, check_finite=True)
    assert eigenvalues.min() > -1e-7
    mean_term = float(np.square(mean-reference_mean).sum())
    covariance_term = float(np.trace(covariance) + np.trace(reference_covariance) -
                            2*np.sqrt(np.maximum(eigenvalues,0)).sum())
    return mean_term + covariance_term


def main(require_complete):
    study.install_infrastructure()
    request, request_hash = study.verify_request()
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb') == request['inception_graph_sha256']
    study.pilot.verify_fit()
    status = read(study.ROOT / 'status.json')
    records = read(study.ROOT / 'results.json') if (study.ROOT / 'results.json').exists() else []
    if require_complete:
        assert status['phase'] == 'complete' and len(records) == 2
    OUT.mkdir(parents=True, exist_ok=True)
    labels = np.load(study.BANK_ROOT / 'labels.npy')
    noise = np.load(study.BANK_ROOT / 'noise.npy', mmap_mode='r')
    assert np.all(np.bincount(labels, minlength=100) == 50)
    assert array_sha(noise) == request['bank']['noise_sha256']
    assert array_sha(labels) == request['bank']['label_sha256']
    with np.load(study.infrastructure.REFERENCE) as ref:
        references = {key: (ref[m].astype(float), ref[c].astype(float))
            for key,m,c in (('pool_3','mu','sigma'),('spatial','mu_s','sigma_s'))}
    roots = {}
    checks, rows = [], []
    for rec in records:
        if not rec['complete']:
            continue
        arm = rec['arm']
        directory = study.ROOT / arm
        assert rec['request_sha256'] == request_hash
        assert rec['samples'] == study.SAMPLES and rec['coverage_verified']
        assert rec['noise_sha256'] == request['bank']['noise_sha256']
        assert rec['label_sha256'] == request['bank']['label_sha256']
        assert sha(rec['sample_path']) == rec['sample_sha256']
        assert sha(directory / 'latents.npy') == rec['latents_sha256']
        assert sha(directory / 'activations.npz') == rec['activations_sha256']
        seen, calls, reader_calls, seconds = set(), 0, 0, 0.
        diagnostic_sum = np.zeros(5)
        for rank in range(study.RANKS):
            d = directory / f'rank{rank}'
            summary = read(d / 'summary.json')
            assert summary['complete'] and summary['request_sha256'] == request_hash
            for entry in summary['files']:
                start, path = entry['start'], d / entry['file']
                assert start not in seen and (start//study.BATCH)%study.RANKS == rank
                seen.add(start)
                assert sha(path) == entry['sha256']
                with np.load(path) as batch:
                    assert str(batch['request_sha256']) == request_hash
                    assert str(batch['noise_sha256']) == array_sha(noise[start:start+study.BATCH])
                    np.testing.assert_array_equal(batch['labels'], labels[start:start+study.BATCH])
                    assert np.isfinite(batch['latents']).all()
                    assert int(batch['prefix_calls']) == int(batch['auxiliary_full_calls']) == 0
                    calls += int(batch['full_calls'])
                    seconds += float(batch['trajectory_seconds'])+float(batch['decode_seconds'])
                    count = batch['block_reader_calls']
                    assert int(batch['reader_calls']) == int(count.sum())
                    reader_calls += int(count.sum())
                    diagnostic_sum += (batch['block_reader_diagnostics']*count[:,None,None]).sum(axis=(0,1))
        assert seen == set(range(0,study.SAMPLES,study.BATCH))
        assert abs(seconds-rec['sum_batch_gpu_seconds']) < 1e-7
        assert calls/(study.SAMPLES/study.BATCH) == rec['full_calls_per_image']
        key = dict(activations_sha256=rec['activations_sha256'], reference_sha256=request['reference_sha256'],
                   fid=rec['fid'], sfid=rec['metrics']['sfid'])
        cache = OUT/f'audit_{arm}.json'
        if cache.exists():
            check = read(cache)
            assert check['key'] == key
        else:
            values = {}
            with np.load(directory/'activations.npz') as features:
                for name in ('pool_3','spatial'):
                    mean,cov = references[name]
                    if name not in roots:
                        vals,vecs = eigh((cov+cov.T)*.5)
                        assert vals.min() > -1e-7
                        roots[name] = (vecs*np.sqrt(np.maximum(vals,0)))@vecs.T
                    assert len(features[name]) == study.SAMPLES
                    values[name] = covariance_fid(features[name],mean,cov,roots[name])
            check = dict(key=key, independent_fid=values['pool_3'],independent_sfid=values['spatial'],
                fid_absolute_error=abs(values['pool_3']-rec['fid']),sfid_absolute_error=abs(values['spatial']-rec['metrics']['sfid']))
            assert check['fid_absolute_error'] < .001 and check['sfid_absolute_error'] < .001
            atomic(cache,check)
        checks.append(dict(arm=arm,batch_coverage_passed=True,**check))
        diag = diagnostic_sum/(reader_calls*study.BATCH) if reader_calls else np.zeros(5)
        rows.append(dict(arm=arm,fid=rec['fid'],sfid=rec['metrics']['sfid'],
            inception_score=rec['metrics']['inception_score'],full_calls_per_image=rec['full_calls_per_image'],
            reader_calls_per_image=reader_calls/(study.SAMPLES/study.BATCH),sum_batch_gpu_seconds=seconds,
            raw_residual_relative_norm=float(diag[0]),predictable_part_relative_norm=float(diag[1]),
            residual_gap_cosine=float(diag[2]),normalized_orthogonal_energy_ratio=float(diag[3]),fallback_fraction=float(diag[4])))
    preflights = [read(p) for p in sorted(study.ROOT.glob('preflight_rank*.json'))]
    assert all(p['passed'] and p['new_noise_norm_identity'] and all(x['exact'] for x in p['both_old_endpoints_exact']) for p in preflights)
    if require_complete:
        assert len(preflights) == study.RANKS
    pictures.OUT, pictures.BANK_ROOT = OUT, study.BANK_ROOT
    examples = pictures.contact_sheet([r for r in records if r['complete']]) if rows else None
    audit = dict(status=status,request_sha256=request_hash,rows=rows,arithmetic_checks=checks,
        preflights=preflights,balanced_class_counts=np.bincount(labels,minlength=100).tolist(),
        fixed_examples=examples,new_noise_confirmation=True,independent_feature_extraction=False,
        independent_training=False,independent_reference=False,research_goal_achieved=False)
    atomic(OUT/'audit.json',audit)
    if rows:
        with (OUT/'metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 小SiT：可预测分歧候选的新噪声5K结果','',f"状态：{status['phase']}；已完成{len(rows)}/2组，每组5000图。",'']
    if len(rows)==2:
        baseline,candidate=rows
        gain=100*(baseline['fid']-candidate['fid'])/baseline['fid']
        if gain>0:
            lines += [f'固定候选在本轮新噪声5K中FID改善{gain:.3f}%，实际采样/解码成本为原生IG的{candidate["sum_batch_gpu_seconds"]/baseline["sum_batch_gpu_seconds"]:.3f}倍。',
                '这是一个模型、一组新噪声的结果；不据此宣称固定点、全模型普适性或论文目标达成。','']
        else:
            lines += [f'候选在新噪声5K上FID恶化{-gain:.3f}%，旧1K的微小改善未获本轮支持。',
                '停止当前固定构造，不围绕新结果调整拟合容量、λ或guidance强度。','']
    lines += ['| 配置 | FID↓ | sFID↓ | IS↑ | Full/图 | batch GPU秒 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['arm']} | {r['fid']:.6f} | {r['sfid']:.6f} | {r['inception_score']:.6f} | {r['full_calls_per_image']:.3f} | {r['sum_batch_gpu_seconds']:.3f} |")
    lines += ['','两组使用完全相同的新噪声和新标签：单CUDA Generator连续B8绘制，noise seed202609981；每类50图，独立标签shuffle seed202609982。',
        '没有重训、改变guide窗口/强度、选择检查点或使用新5K拟合参数。原生IG也在本轮重新采样；采样/解码成本包含候选的实际逐查询诊断。',
        '固定候选只增加6160个线性系数和标准化统计，使用原生一次Full中的浅层特征；主干、强头、旧weak头不更新。',
        '具体构造：记D=S−W，C(H)为仅从浅层条件特征H拟合D的仿射读出，R=D−C(H)，R̂=||D||R/||R||；新速度为S+αR̂。',
        '全图范数退化时的回退规则与1K一致；本轮实际没有触发回退。这里先形成新方向，再保持原gap长度。',
        '此处并未将“可预测”解释成有害或多余的真实信息；它是固定仿射函数族的操作性分解。','',
        '旧1K原生/候选为65.139320/64.972278，平行分量对照65.567291；不要直接用1K和5K的绝对FID作改善比较。',
        '新5K未重复parallel组，因此没有独立确认该方向因果对照。真实ADM参考、模型训练与读出拟合资产均共用；只更换质量采样的噪声和标签。','',
        '在候选5K的实际查询上，R与D的平均余弦约0.941，等长残差的正交能量比例约11.33%。在出现非零正交分量的状态上，该向量场不能由单一scalar IG系数精确表示。',
        '这不证明其优于重新优化后的普通IG强度曲线，也不等于生成分布不可由别的采样方案近似；质量结论仍只限本次固定配置。',
        '剩余部分更难被这个便宜读出预测，并不意味着它更接近真实数据、具备更多Shannon信息或自动具有guidance价值。当前的正向FID是实验结果，不是回归正交性定理的推论。','',
        '四卡在正式5K前分别复现旧原生与旧候选的首批latent，均逐元素相同。新bank的norm恒等式通过；每组625个batch覆盖、输入、标签、权重、源码与评估图身份均核对。',
        '独立FP64核算使用对称协方差平方根的特征空间公式，与ADM实现不同；仍共用缓存Inception特征，不是独立特征提取。','',
        '[冻结5K协议](SMALL_SIT_PREDICTABLE_GAP_CONFIRMATION_PROTOCOL_20260909_ZH.md) · '
        '[1K探索结果](SMALL_SIT_PREDICTABLE_GAP_RESULTS_20260909_ZH.md) · '
        '[5K指标](data/small_sit_predictable_gap_confirmation_20260909/metrics.csv) · '
        '[审计](data/small_sit_predictable_gap_confirmation_20260909/audit.json)','',
        f'原始产物：`{study.ROOT}`；请求SHA256：`{request_hash}`。','',
        '固定最前8个新输入，无质量选择：','','![5K配对首8张](data/small_sit_predictable_gap_confirmation_20260909/paired_first8.png)']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=status['phase'],completed=len(rows),rows=rows,report=str(REPORT))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true')
    main(parser.parse_args().require_complete)
