"""Audit the two fixed precision-balance trials and their reused IG baseline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments import analyze_small_sit_carrier_flow_20260909 as pictures
from experiments import small_sit_precision_balance_20260909 as study
from experiments.analyze_official_sit_pfr_symmetric_20260909 import fid_components
from experiments.lifting_scale_sweep_20260909 import array_sha, atomic, read, sha

OUT = study.WORK / 'docs/data/small_sit_precision_balance_20260909'
REPORT = study.WORK / 'docs/SMALL_SIT_PRECISION_BALANCE_RESULTS_20260909_ZH.md'
TITLES = dict(ig_restarted='普通IG，复用', precision_rank1='一维精度平衡', precision_rank2='二维精度平衡')


def main(require_complete):
    study.install_infrastructure()
    request, request_hash = study.infrastructure.verify_request()
    root = study.ROOT
    status = read(root / 'status.json')
    new = read(root / 'results.json') if (root / 'results.json').exists() else []
    if require_complete:
        assert status['phase'] == 'complete' and len(new) == len(study.ARMS)
    records = [request['baseline']] + new
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(study.infrastructure.REFERENCE) as ref:
        mu, cov = ref['mu'].astype(float), ref['sigma'].astype(float)
        mu_s, cov_s = ref['mu_s'].astype(float), ref['sigma_s'].astype(float)
    noise = np.load(study.infrastructure.BANK_ROOT / 'noise.npy', mmap_mode='r')
    labels = np.load(study.infrastructure.BANK_ROOT / 'labels.npy')
    audit, rows = [], []
    for rec in records:
        if not rec['complete']:
            continue
        arm = rec['arm']
        d = Path(rec['sample_path']).parent
        expected_request = request['parent_request_sha256'] if arm == 'ig_restarted' else request_hash
        assert rec['request_sha256'] == expected_request
        assert rec['noise_sha256'] == request['bank']['noise_sha256']
        assert rec['label_sha256'] == request['bank']['label_sha256']
        assert rec['coverage_verified'] and rec['samples'] == study.SAMPLES
        assert sha(rec['sample_path']) == rec['sample_sha256']
        assert sha(d / 'latents.npy') == rec['latents_sha256']
        assert sha(d / 'activations.npz') == rec['activations_sha256']
        seen, modes = set(), np.zeros(3, np.int64)
        weighted_diag = np.zeros(6)
        max_mean_alpha = 0.
        for rank in range(study.RANKS):
            directory = d / f'rank{rank}'
            summary = read(directory / 'summary.json')
            assert summary['complete'] and summary['request_sha256'] == expected_request
            for entry in summary['files']:
                start, path = entry['start'], directory / entry['file']
                assert start not in seen and (start // study.BATCH) % study.RANKS == rank
                assert sha(path) == entry['sha256']
                seen.add(start)
                with np.load(path) as b:
                    assert str(b['request_sha256']) == expected_request
                    assert str(b['noise_sha256']) == array_sha(noise[start:start + study.BATCH])
                    np.testing.assert_array_equal(b['labels'], labels[start:start + study.BATCH])
                    if arm != 'ig_restarted':
                        counts, diag = b['block_mode_counts'], b['block_response_means']
                        modes += counts.sum(axis=(0, 1))
                        weighted_diag += (diag * counts.sum(axis=-1)[..., None]).sum(axis=(0, 1))
                        max_mean_alpha = max(max_mean_alpha, float(diag[:, :, 1].max()))
        assert seen == set(range(0, study.SAMPLES, study.BATCH))
        key = dict(activations_sha256=rec['activations_sha256'], reference_sha256=request['reference_sha256'],
                   fid=rec['fid'], sfid=rec['metrics']['sfid'])
        cache = OUT / f'audit_{arm}.json'
        if cache.exists():
            check = read(cache)
            assert check['key'] == key
        else:
            with np.load(d / 'activations.npz') as acts:
                fm, fc = fid_components(acts['pool_3'].astype(float), mu, cov)
                sm, sc = fid_components(acts['spatial'].astype(float), mu_s, cov_s)
            check = dict(key=key, independent_fid=fm + fc, independent_sfid=sm + sc,
                fid_absolute_error=abs(fm + fc - rec['fid']), sfid_absolute_error=abs(sm + sc - rec['metrics']['sfid']))
            assert check['fid_absolute_error'] < .001 and check['sfid_absolute_error'] < .001, check
            atomic(cache, check)
        audit.append(dict(arm=arm, coverage_passed=True, **check))
        mean_diag = weighted_diag / modes.sum() if modes.sum() else np.zeros(6)
        rows.append(dict(arm=arm, fid=rec['fid'], sfid=rec['metrics']['sfid'],
            inception_score=rec['metrics']['inception_score'], full_calls_per_image=rec['full_calls_per_image'],
            auxiliary_full_calls_per_image=rec['auxiliary_full_calls_per_image'],
            sum_batch_gpu_seconds=rec['sum_batch_gpu_seconds'],
            mode_counts=modes.tolist(), mode_fractions=(modes / modes.sum()).tolist() if modes.sum() else [0., 0., 0.],
            query_weighted_effective_alpha=float(mean_diag[1]),
            query_weighted_orthogonal_relative=float(mean_diag[2]),
            query_weighted_strong_antisymmetry=float(mean_diag[3]),
            query_weighted_weak_antisymmetry=float(mean_diag[4]),
            maximum_batch_block_image_mean_alpha=max_mean_alpha))
    sensitivity = []
    for p in sorted(root.glob('preflight_rank*.json')):
        preflight = read(p)
        assert preflight['passed'] and preflight['ordinary_ig_latents_exact']
        for item in preflight['finite_difference_checks']:
            for i, value in enumerate(item['relative_output_difference']):
                sensitivity.append(dict(rank=preflight['rank'], batch_offset=i, t=item['t'],
                    relative_output_difference=value, mode1=item['selected_rank_at_fd1'][i],
                    mode_half=item['selected_rank_at_fd_half'][i]))
    pictures.OUT = OUT
    fixed_examples = pictures.contact_sheet([rec for rec in records if rec['complete']])
    value = dict(status=status, request_sha256=request_hash, rows=rows, arithmetic_checks=audit,
        finite_difference_sensitivity=sensitivity, fixed_examples=fixed_examples,
        arithmetic_only_not_independent_features=True, independent_confirmation=False, research_goal_achieved=False)
    derivative = None
    if (root / 'derivative_audit.json').exists():
        derivative = read(root / 'derivative_audit.json')
        assert derivative['request_sha256'] == request_hash
        value['derivative_audit_path'] = str(root / 'derivative_audit.json')
        value['derivative_audit_sha256'] = sha(root / 'derivative_audit.json')
        value['autodiff_admissibility'] = dict(states=len(derivative['rows']),
            strong_positive=sum(min(x['balance']['autodiff']['strong_eigenvalues']) > 1e-6 for x in derivative['rows']),
            weak_positive=sum(min(x['balance']['autodiff']['weak_eigenvalues']) > 1e-6 for x in derivative['rows']),
            combination_positive=sum(min(x['balance']['autodiff']['combination_eigenvalues']) > 1e-6 for x in derivative['rows']))
    atomic(OUT / 'audit.json', value)
    with (OUT / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 小SiT：精度平衡候选的真实生成结果', '',
        f"状态：{status['phase']}；新配置完整质量结果{sum(rec['complete'] for rec in new)}/2；同分段普通IG复用。", '',
        '这轮从FSG的信息承载问题出发，测试局部后验精度加权的残差平衡。',
        '数学动机、可表达性限制与非零头差的等式见[理论记录](IG_FIXED_POINT_MESSAGE_GEOMETRY_20260909_ZH.md)，',
        '实际低维近似、数值条件和失败处理见[冻结协议](SMALL_SIT_PRECISION_BALANCE_PROTOCOL_20260909_ZH.md)。', '',
        '| 配置 | FID↓ | sFID↓ | IS↑ | Full/图 | batch GPU秒 | 相对IG成本 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    baseline = rows[0]
    for row in rows:
        lines.append(f"| {TITLES[row['arm']]} | {row['fid']:.6f} | {row['sfid']:.6f} | {row['inception_score']:.6f} | "
            f"{row['full_calls_per_image']:.3f} | {row['sum_batch_gpu_seconds']:.2f} | "
            f"{row['sum_batch_gpu_seconds'] / baseline['sum_batch_gpu_seconds']:.3f}x |")
    if status['phase'] == 'paused_after_numerical_review':
        lines += ['', f"在数值/成本复核后中止二维配置，保留{status['partial_images']}张完整落盘样本；不计算或报告部分样本FID。",
            '一维配置FID明显恶化，二维前几批还出现约8000次Full/图和极端系数放大。此实现不具备继续补满质量采样的价值。',
            '原冻结代码和数值规则没有修改；人工中止记录、原状态与完整batch保留。这不是声称未完成的二维FID必然更差。']
    if len(rows) == 3:
        better = [row for row in rows[1:] if row['fid'] < baseline['fid']]
        if not better:
            lines += ['', '两个候选均未改善普通IG的FID；本轮没有质量依据支持扩扫或独立5K。']
        else:
            lines += ['', '至少一个候选在这组探索性1K上改善FID。该结果尚不能区分调强度、额外计算、数值实现与额外响应信息的贡献，也不构成独立确认。']
    lines += ['', '相同模型、noise、标签、主Dopri5精度与五段边界；首批普通IG latent逐元素复现，zero-gamma退化检查通过。',
        '原计划每个新配置1K，实际完成数以上方状态为准；输入来自已有探索bank，没有训练新参数。全部成本为采样与解码的batch耗时之和，不含预检和ADM评价。', '',
        '| 候选 | 回退普通IG比例 | 采用一维比例 | 采用二维比例 | 查询加权有效alpha | 正交修正/普通修正 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows[1:]:
        f0, f1, f2 = row['mode_fractions']
        lines.append(f"| {TITLES[row['arm']]} | {f0:.4%} | {f1:.4%} | {f2:.4%} | "
            f"{row['query_weighted_effective_alpha']:.6f} | {row['query_weighted_orthogonal_relative']:.6f} |")
    lines += ['', '上述读数覆盖自适应求解器的所有活动RHS查询，包括被拒绝的试探步；不是均匀时间采样，也不是只统计接受的轨迹。',
        '因此大系数可能出现在求解器拒绝的状态上；NFE和实际耗时仍完整计入。',
        '投影Jacobian取对称部分、近奇异时回退，属于预先固定的近似规则，不能据此认证真实后验合法性。']
    if sensitivity:
        errors = np.array([item['relative_output_difference'] for item in sensitivity])
        lines += ['', f"真实网络预检的{len(errors)}个状态/样本中，将差分位移从1%减为0.5%后，输出相对变化中位数{np.median(errors):.6f}、最大值{errors.max():.6f}。",
            '这是实际数值敏感性，入口与零强度检查通过并不意味着局部响应估计稳定。现有质量结果属于固定1%实现，不能当作精确Jacobian或精确Gaussian后验的结果。']
    if derivative:
        checks = value['autodiff_admissibility']
        lines += ['', f"采样停止后，对相同的{checks['states']}个检查状态做一阶反向自动微分。固定1%差分定义的同一组基，避免把基变动与导数误差混在一起。",
            f"Strong/Weak的对称投影响应分别在{checks['strong_positive']}/{checks['states']}和{checks['weak_positive']}/{checks['states']}个检查点正定；组合矩阵只有{checks['combination_positive']}/{checks['states']}个正定。",
            '因此不适定性不能全部归因于有限差分精度。这个局部Gaussian近似的可归一化前提经常不成立；这不等于证明原始神经模型对应的真实密度不可归一化。',
            f"1%差分相对自动微分的矩阵误差中位数：Strong {derivative['summary']['strong']['fd1']['median']:.6f}、Weak {derivative['summary']['weak']['fd1']['median']:.6f}；",
            f"最大误差分别为{derivative['summary']['strong']['fd1']['maximum']:.6f}、{derivative['summary']['weak']['fd1']['maximum']:.6f}。原始逐状态记录见[导数审计](data/small_sit_precision_balance_20260909/derivative_audit.json)。"]
    for rec in new:
        if not rec['complete']:
            lines += ['', f"失败配置{rec['arm']}：{json.dumps(rec, ensure_ascii=False)}"]
    lines += ['', 'ADM缓存特征上的FID/sFID经独立FP64样本空间公式核验；不是独立特征提取或独立数据确认。',
        f'请求SHA256：`{request_hash}`；原始资产：`{root}`。', '',
        '[指标](data/small_sit_precision_balance_20260909/metrics.csv) · [审计与数值敏感性](data/small_sit_precision_balance_20260909/audit.json)', '',
        '输入索引0–7固定展示，未按效果挑图：', '',
        '![配对生成](data/small_sit_precision_balance_20260909/paired_first8.png)']
    REPORT.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=status['phase'], completed=len(rows) - 1, report=str(REPORT))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    main(parser.parse_args().require_complete)
