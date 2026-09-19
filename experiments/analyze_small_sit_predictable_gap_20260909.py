"""Audit the weak-feature residualization experiment and its fixed controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from experiments import small_sit_predictable_gap_20260909 as study
from experiments import analyze_small_sit_carrier_flow_20260909 as pictures
from experiments.analyze_official_sit_pfr_symmetric_20260909 import fid_components
from experiments.lifting_scale_sweep_20260909 import array_sha, atomic, read, sha

OUT = study.WORK / 'docs/data/small_sit_predictable_gap_20260909'
REPORT = study.WORK / 'docs/SMALL_SIT_PREDICTABLE_GAP_RESULTS_20260909_ZH.md'
TITLES = dict(ig_restarted='原生IG，复用', residual_raw='去除可预测部分',
              residual_norm='残差与原gap等长', parallel_norm='仅保留平行分量')


def main(require_complete):
    study.install_infrastructure()
    request, request_hash = study.infrastructure.verify_request()
    fit_request = study.verify_fit()
    assert sha(study.ROOT / 'fit_request.json') == request['fit_request_sha256']
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb') == request['inception_graph_sha256']
    fit = read(study.ROOT / 'fit.json')
    assert sha(fit['checkpoint']) == fit['sha256']
    assert sha(study.ROOT / 'fit_equations.npz') == fit['equation_sha256']
    with np.load(study.ROOT / 'fit_equations.npz') as equations:
        independent = solve(equations['gram'] + equations['penalty'], equations['target'], assume_a='pos')
        regression_error = float(np.max(np.abs(independent - equations['coefficient'])))
        assert regression_error < 1e-10
    status = read(study.ROOT / 'status.json')
    results = read(study.ROOT / 'results.json') if (study.ROOT / 'results.json').exists() else []
    if require_complete:
        assert status['phase'] == 'complete' and len(results) == len(study.ARMS)
    records = [request['baseline']] + results
    OUT.mkdir(parents=True, exist_ok=True)
    noise = np.load(study.infrastructure.BANK_ROOT / 'noise.npy', mmap_mode='r')
    labels = np.load(study.infrastructure.BANK_ROOT / 'labels.npy')
    with np.load(study.infrastructure.REFERENCE) as ref:
        mu, cov = ref['mu'].astype(float), ref['sigma'].astype(float)
        mu_s, cov_s = ref['mu_s'].astype(float), ref['sigma_s'].astype(float)
    rows, checks = [], []
    for rec in records:
        if not rec['complete']:
            continue
        arm = rec['arm']
        directory = Path(rec['sample_path']).parent
        expected = request['parent_request_sha256'] if arm == 'ig_restarted' else request_hash
        assert rec['request_sha256'] == expected
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
            assert summary['complete'] and summary['request_sha256'] == expected
            for entry in summary['files']:
                start, path = entry['start'], d / entry['file']
                assert start not in seen and (start // study.BATCH) % study.RANKS == rank
                seen.add(start)
                assert sha(path) == entry['sha256']
                with np.load(path) as batch:
                    assert str(batch['request_sha256']) == expected
                    assert str(batch['noise_sha256']) == array_sha(noise[start:start + study.BATCH])
                    np.testing.assert_array_equal(batch['labels'], labels[start:start + study.BATCH])
                    assert np.isfinite(batch['latents']).all()
                    assert int(batch['prefix_calls']) == int(batch['auxiliary_full_calls']) == 0
                    calls += int(batch['full_calls'])
                    seconds += float(batch['trajectory_seconds']) + float(batch['decode_seconds'])
                    if arm != 'ig_restarted':
                        counts = batch['block_reader_calls']
                        reader_calls += int(counts.sum())
                        assert int(batch['reader_calls']) == int(counts.sum())
                        diagnostic_sum += (batch['block_reader_diagnostics'] * counts[:, None, None]).sum(axis=(0, 1))
        assert seen == set(range(0, study.SAMPLES, study.BATCH))
        assert calls / (study.SAMPLES / study.BATCH) == rec['full_calls_per_image']
        assert abs(seconds - rec['sum_batch_gpu_seconds']) < 1e-7
        key = dict(activations_sha256=rec['activations_sha256'], reference_sha256=request['reference_sha256'],
                   fid=rec['fid'], sfid=rec['metrics']['sfid'])
        cache = OUT / f'audit_{arm}.json'
        if cache.exists():
            check = read(cache)
            assert check['key'] == key
        else:
            with np.load(directory / 'activations.npz') as acts:
                assert len(acts['pool_3']) == len(acts['spatial']) == study.SAMPLES
                fm, fc = fid_components(acts['pool_3'].astype(float), mu, cov)
                sm, sc = fid_components(acts['spatial'].astype(float), mu_s, cov_s)
            check = dict(key=key, independent_fid=fm + fc, independent_sfid=sm + sc,
                fid_absolute_error=abs(fm + fc - rec['fid']), sfid_absolute_error=abs(sm + sc - rec['metrics']['sfid']))
            assert check['fid_absolute_error'] < .001 and check['sfid_absolute_error'] < .001
            atomic(cache, check)
        checks.append(dict(arm=arm, coverage_passed=True, **check))
        diag = diagnostic_sum / (reader_calls * study.BATCH) if reader_calls else np.zeros(5)
        rows.append(dict(arm=arm, fid=rec['fid'], sfid=rec['metrics']['sfid'],
            inception_score=rec['metrics']['inception_score'], full_calls_per_image=rec['full_calls_per_image'],
            reader_calls_per_image=reader_calls / (study.SAMPLES / study.BATCH),
            sum_batch_gpu_seconds=rec['sum_batch_gpu_seconds'], raw_residual_relative_norm=float(diag[0]),
            predictable_part_relative_norm=float(diag[1]), residual_gap_cosine=float(diag[2]),
            normalized_orthogonal_energy_ratio=float(diag[3]), fallback_fraction=float(diag[4])))
    preflights = [read(p) for p in sorted(study.ROOT.glob('preflight_rank*.json'))]
    assert all(p['passed'] and p['ordinary_ig_latents_exact'] for p in preflights)
    if require_complete:
        assert len(preflights) == study.RANKS
    timing_ratios = {arm: float(np.median([p['per_query_seconds'][arm] / p['per_query_seconds']['native']
        for p in preflights])) for arm in study.ARMS} if preflights else {}
    amendment_path = study.ROOT / 'amendments/01_projection_precision/amendment.json'
    amendment = read(amendment_path) if amendment_path.exists() else None
    if amendment is not None:
        assert amendment['unchanged_projection_sha256'] == fit['sha256']
        assert amendment['quality_batches_before_fix'] == 0
    pictures.OUT = OUT
    examples = pictures.contact_sheet([r for r in records if r['complete']])
    audit = dict(status=status, request_sha256=request_hash, fit=fit, regression_coefficient_max_error=regression_error,
        source_asset_cache_hashes_passed=True, arithmetic_checks=checks, rows=rows, preflights=preflights,
        fixed_examples=examples, fixed_rhs_timing_ratios=timing_ratios, precision_amendment=amendment,
        independent_confirmation=False, independent_feature_extraction=False,
        research_goal_achieved=False)
    atomic(OUT / 'audit.json', audit)
    with (OUT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = rows[0]
    lines = ['# 小SiT：浅层可预测分歧的质量结果', '',
        f"状态：{status['phase']}；完成的新配置{len(rows)-1}/3，另复用完整原生IG基线。", '']
    if len(rows) == 4:
        raw, norm, parallel = rows[1:]
        if norm['fid'] < baseline['fid'] and norm['fid'] < parallel['fid']:
            lines += ['等长残差方向在本轮探索1K上优于普通IG和仅平行分量对照，形成初步方向信号。',
                '尚不能据此认定总体改善、解释了深度信息或实现论文目标。', '']
        elif all(r['fid'] >= baseline['fid'] for r in rows[1:]):
            lines += ['三个新配置均未改善普通IG，当前固定构造的质量假说未获支持；停止该构造，不继续调λ或归一化。', '']
        else:
            lines += ['部分配置改变了FID，但未同时建立等长残差优于普通IG与平行分量对照的方向优势。', '']
    lines += ['| 配置 | FID↓ | sFID↓ | IS↑ | Full/图 | batch GPU秒 | 相对旧IG成本 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {TITLES[r['arm']]} | {r['fid']:.6f} | {r['sfid']:.6f} | {r['inception_score']:.6f} | "
            f"{r['full_calls_per_image']:.3f} | {r['sum_batch_gpu_seconds']:.3f} | {r['sum_batch_gpu_seconds']/baseline['sum_batch_gpu_seconds']:.3f}x |")
    lines += ['', '同一1000个噪声和类别、B8、原生网格、Dopri5容差与解码器；每个RHS只有一次原生主干调用。',
        '新条件增加6160个线性系数及标准化统计量，强头没有改变，也没有读取深层特征作为回归输入。',
        '总成本包含实际采样、逐查询诊断与解码，另有拟合、加载、预检和评价成本。旧基线不是同期吞吐复测。', '',
        f"在已缓存的131072个训练token上，FP64拟合及方程/权重保存耗时{fit['seconds']:.4f}秒，不含cache加载和后续验证。",
        '原cache来自8192训练图，一次前向采集成本5.5873秒，属于前一轮已支付并复用的成本；不把缓存视为免费外部资源。',
        f"留出16384个token上，残差能量是原gap的{fit['metrics'][1]['residual_energy_ratio']:.6f}倍，减少{100*(1-fit['metrics'][1]['residual_energy_ratio']):.3f}%。",
        '这只是受限仿射预测能力，不是有害成分的比例，也不是生成质量证据。', '',
        '| 配置 | R/D长度比 | C/D长度比 | cos(R,D) | 等长残差正交能量比 | 回退比例 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows[1:]:
        lines.append(f"| {TITLES[r['arm']]} | {r['raw_residual_relative_norm']:.6f} | {r['predictable_part_relative_norm']:.6f} | "
            f"{r['residual_gap_cosine']:.6f} | {r['normalized_orthogonal_energy_ratio']:.6f} | {r['fallback_fraction']:.8f} |")
    lines += ['', '上表按所有活动RHS查询加权，包括自适应积分拒绝的试探；各配置走自己的状态，不能把跨行差异当成同状态因果估计。',
        'parallel组只检验norm方向新增正交分量的价值；原始残差改善若存在，也可能来自有效强度变化。', '',
        f"SciPy从保存的正规方程独立求解，系数最大差{regression_error:.3e}。真实模型上GPU读出与NumPy矩阵乘法、",
        '弱头原生线性读出、norm/parallel几何恒等式、零α以及禁用新操作的原生IG首批latent均通过检查。',
        '所有配置各125个batch覆盖、噪声标签、模型/源码/cache/投影参数/评价图哈希均核对；FID和sFID用相同特征上的FP64公式复算。',
        '这不是独立特征提取，重复探索bank也不是独立样本确认。', '',
        '深层特征是浅层特征的确定性后续计算；这里的“可预测”仅相对于固定、廉价、局部的仿射读出族。',
        '移除可预测项没有质量保证，也不是新的残差化原理。既有文献和旧失败对照见冻结协议。', '',
        '[冻结协议](SMALL_SIT_PREDICTABLE_GAP_PROTOCOL_20260909_ZH.md) · '
        '[指标](data/small_sit_predictable_gap_20260909/metrics.csv) · '
        '[完整审计](data/small_sit_predictable_gap_20260909/audit.json)', '',
        f'原始产物：`{study.ROOT}`；采样请求SHA256：`{request_hash}`。', '',
        '固定最前8张，无质量筛选：', '', '![固定配对样本](data/small_sit_predictable_gap_20260909/paired_first8.png)']
    if timing_ratios:
        where = lines.index('总成本包含实际采样、逐查询诊断与解码，另有拟合、加载、预检和评价成本。旧基线不是同期吞吐复测。') + 1
        lines[where:where] = ['固定同状态预热后30次RHS计时，4个rank的相对耗时中位数：' +
            '，'.join(f'{arm} {timing_ratios[arm]:.4f}x' for arm in study.ARMS) + '。该计时包含实际诊断，不是端到端重测。']
    if amendment is not None:
        where = lines.index('弱头原生线性读出、norm/parallel几何恒等式、零α以及禁用新操作的原生IG首批latent均通过检查。') + 1
        lines[where:where] = ['首次预检中TF32新读出与高精度NumPy的最大误差约0.00038–0.00042，超过预设0.0003；正式质量batch为0。',
            '仅将新线性读出的乘法改为完整FP32，并立即恢复主干的TF32设置，原拟合系数不变；新源与修订重新冻结后全部预检通过。',
            '原请求、失败日志、修订前后源码和系数身份保留在原始目录amendments/01_projection_precision。']
    REPORT.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=status['phase'], completed=len(rows)-1, rows=rows, report=str(REPORT))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    main(parser.parse_args().require_complete)
