"""Check paired outputs, training identity and actual cost of the cheap readers."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments import small_sit_feedback_reader_20260909 as study
from experiments import analyze_small_sit_carrier_flow_20260909 as pictures
from experiments.analyze_official_sit_pfr_symmetric_20260909 import fid_components
from experiments.lifting_scale_sweep_20260909 import array_sha, atomic, read, sha

OUT = study.WORK / 'docs/data/small_sit_feedback_reader_20260909'
REPORT = study.WORK / 'docs/SMALL_SIT_FEEDBACK_READER_RESULTS_20260909_ZH.md'
TITLES = dict(ig_restarted='普通IG，复用', strong_only='只反馈Strong', strong_gap='反馈Strong与分歧')


def main(require_complete):
    study.install_infrastructure()
    request, request_hash = study.infrastructure.verify_request()
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb') == request['inception_graph_sha256']
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
    rows, audits = [], []
    for rec in records:
        if not rec['complete']:
            continue
        arm = rec['arm']
        d = Path(rec['sample_path']).parent
        expected_request = request['parent_request_sha256'] if arm == 'ig_restarted' else request_hash
        assert rec['request_sha256'] == expected_request
        assert rec['coverage_verified'] and rec['samples'] == study.SAMPLES
        assert rec['noise_sha256'] == request['bank']['noise_sha256']
        assert rec['label_sha256'] == request['bank']['label_sha256']
        assert sha(rec['sample_path']) == rec['sample_sha256']
        assert sha(d / 'latents.npy') == rec['latents_sha256']
        assert sha(d / 'activations.npz') == rec['activations_sha256']
        seen, reader_calls = set(), 0
        sums = np.zeros(4)
        total_full, total_seconds = 0, 0.
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
                    assert np.isfinite(b['latents']).all()
                    total_full += int(b['full_calls'])
                    total_seconds += float(b['trajectory_seconds']) + float(b['decode_seconds'])
                    if arm != 'ig_restarted':
                        count = b['block_reader_calls']
                        reader_calls += int(count.sum())
                        sums += (b['block_reader_diagnostics'] * count[:, None, None]).sum(axis=(0, 1))
                        assert int(b['auxiliary_full_calls']) == 0 and int(b['prefix_calls']) == 0
                        assert int(b['reader_calls']) == int(count.sum())
        assert seen == set(range(0, study.SAMPLES, study.BATCH))
        assert abs(total_seconds - rec['sum_batch_gpu_seconds']) < 1e-7
        assert total_full / (study.SAMPLES / study.BATCH) == rec['full_calls_per_image']
        key = dict(activations_sha256=rec['activations_sha256'], reference_sha256=request['reference_sha256'],
                   fid=rec['fid'], sfid=rec['metrics']['sfid'])
        cache = OUT / f'audit_{arm}.json'
        if cache.exists():
            check = read(cache)
            assert check['key'] == key
        else:
            with np.load(d / 'activations.npz') as acts:
                assert len(acts['pool_3']) == len(acts['spatial']) == study.SAMPLES
                fm, fc = fid_components(acts['pool_3'].astype(float), mu, cov)
                sm, sc = fid_components(acts['spatial'].astype(float), mu_s, cov_s)
            check = dict(key=key, independent_fid=fm + fc, independent_sfid=sm + sc,
                fid_absolute_error=abs(fm + fc - rec['fid']), sfid_absolute_error=abs(sm + sc - rec['metrics']['sfid']))
            assert check['fid_absolute_error'] < .001 and check['sfid_absolute_error'] < .001
            atomic(cache, check)
        audits.append(dict(arm=arm, coverage_passed=True, **check))
        diag = sums / (reader_calls * study.BATCH) if reader_calls else np.zeros(4)
        rows.append(dict(arm=arm, fid=rec['fid'], sfid=rec['metrics']['sfid'],
            inception_score=rec['metrics']['inception_score'], full_calls_per_image=rec['full_calls_per_image'],
            auxiliary_full_calls_per_image=rec['auxiliary_full_calls_per_image'],
            reader_calls_per_image=reader_calls / (study.SAMPLES / study.BATCH),
            sum_batch_gpu_seconds=rec['sum_batch_gpu_seconds'],
            query_weighted_guided_change_relative_rms=float(diag[0]),
            query_weighted_gap_relative_rms=float(diag[1]),
            query_weighted_strong_change_rms=float(diag[2]),
            query_weighted_weak_change_rms=float(diag[3])))
    training = study.verify_training()
    assert sha(root / 'training_request.json') == request['training_request_sha256']
    train_ids, val_ids = np.load(root / 'train_indices.npy'), np.load(root / 'validation_indices.npy')
    assert len(set(train_ids.tolist()).intersection(val_ids.tolist())) == 0
    fits = [read(root / f'training_{arm}.json') for arm in study.ARMS]
    for fit in fits:
        assert fit['complete'] and fit['steps'] == study.TRAIN_STEPS
        assert sha(fit['checkpoint']) == fit['sha256']
    collection = read(root / 'collection.json')
    for rec in collection['records']:
        assert sha(rec['path']) == rec['sha256']
    per_query = []
    for path in sorted(root.glob('preflight_rank*.json')):
        preflight = read(path)
        assert preflight['passed'] and preflight['ordinary_ig_latents_exact']
        seconds = preflight['per_query_seconds']
        per_query.append(dict(rank=preflight['rank'], seconds=seconds,
            ratios={arm: seconds[arm] / seconds['native'] for arm in study.ARMS}))
    pictures.OUT = OUT
    images = pictures.contact_sheet([rec for rec in records if rec['complete']])
    cost_audit = read(root / 'cost_audit.json') if (root / 'cost_audit.json').exists() else None
    if cost_audit is not None:
        assert cost_audit['request_sha256'] == request_hash
        assert all(cost_audit['pure_outputs_equal_instrumented'].values())
        assert cost_audit['source_sha256'] == sha(study.WORK / 'experiments/audit_small_sit_feedback_reader_cost_20260909.py')
        atomic(OUT / 'cost_audit.json', cost_audit)
    value = dict(status=status, request_sha256=request_hash, rows=rows, arithmetic_checks=audits,
        training_request_sha256=request['training_request_sha256'], training_index_disjoint=True,
        collection=collection, training=fits, per_query_timing=per_query, isolated_cost_audit=cost_audit, fixed_examples=images,
        new_independent_confirmation=False, independent_features=False, research_goal_achieved=False)
    atomic(OUT / 'audit.json', value)
    with (OUT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = rows[0]
    lines = ['# 小SiT：低成本反馈读出结果', '',
        f"状态：{status['phase']}；新配置完整1K结果{len(rows)-1}/2；普通IG完整配对基线复用。", '']
    if len(rows) == 3:
        only, gap = rows[1:]
        if gap['fid'] < baseline['fid'] and gap['fid'] < only['fid']:
            lines += ['显式读取分歧的候选在本轮探索1K上同时优于普通IG和Strong-only读出。',
                '这是初步质量信号，尚需结合实际成本、替代解释和独立样本确认，不能据此宣称固定点理论或研究目标成功。', '']
        elif gap['fid'] < baseline['fid']:
            lines += ['候选优于普通IG，但未超过Strong-only读出；本轮没有建立显式分歧的增量质量价值。', '']
        else:
            lines += ['显式读取分歧的候选未改善普通IG的FID；本轮未建立这项低成本构造的生成质量优势。', '']
    lines += ['| 配置 | FID↓ | sFID↓ | IS↑ | Full/图 | 双读出/图 | batch GPU秒 | 相对IG成本 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {TITLES[row['arm']]} | {row['fid']:.6f} | {row['sfid']:.6f} | {row['inception_score']:.6f} | "
            f"{row['full_calls_per_image']:.3f} | {row['reader_calls_per_image']:.3f} | {row['sum_batch_gpu_seconds']:.2f} | "
            f"{row['sum_batch_gpu_seconds']/baseline['sum_batch_gpu_seconds']:.3f}x |")
    lines += ['', '每个RHS仍只做一次原生主干；额外Full/prefix均为0。实际总NFE受自适应求解器影响。',
        '同一噪声标签、B8、FP32/TF32、分段Dopri5及原生gamma形状；额外读出只在guidance活动区间启用。',
        '新方案同时改变活动区间的Strong基准和Strong–Weak差值；对照考察显式读取差值的增量。',
        '总成本为全部batch采样与解码之和，不含训练、加载、预检和ADM评价。旧基线并非同一时刻重跑的严格吞吐基准。', '',
        f"每个配置新增{fits[0]['parameters']:,}参数；固定8192训练图、1024互斥验证图，每图16patch，1000优化步。",
        f"一次特征采集与缓存用时{collection['seconds']:.2f}秒；两组拟合与过程验证分别{fits[0]['seconds']:.2f}、{fits[1]['seconds']:.2f} GPU秒。",
        '上述时间区间包含相应CPU调度／同步；加载和请求哈希开销另计。训练使用真实velocity，固定末步EMA，没有按验证或FID选检查点。', '',
        '| 配置 | 验证Strong MSE | 验证Weak MSE | 验证guided MSE | 验证gap均方 |',
        '|---|---:|---:|---:|---:|']
    first = fits[0]['rows'][-1]['validation']
    lines.append(f"| 原生读出 | {first['native_strong_mse']:.6f} | {first['native_weak_mse']:.6f} | "
                 f"{first['native_guided_mse']:.6f} | {first['native_gap_mean_square']:.6f} |")
    for fit in fits:
        v = fit['rows'][-1]['validation']
        lines.append(f"| {TITLES[fit['arm']]} | {v['strong_mse']:.6f} | {v['weak_mse']:.6f} | {v['guided_mse']:.6f} | {v['gap_mean_square']:.6f} |")
    lines += ['', '验证MSE来自真实图加噪的固定patch集合，不能直接替代生成FID；分歧缩小也不是成功判据。', '',
        '| 配置 | 单RHS耗时比，中位数 | 引导场改变量/原场RMS | 新旧gap RMS比 |',
        '|---|---:|---:|---:|']
    for row in rows[1:]:
        ratio = np.median([r['ratios'][row['arm']] for r in per_query])
        lines.append(f"| {TITLES[row['arm']]} | {ratio:.4f}x | {row['query_weighted_guided_change_relative_rms']:.6f} | "
                     f"{row['query_weighted_gap_relative_rms']:.6f} |")
    lines += ['', '单RHS计时使用固定同一batch-state、预热后30次调用；不是端到端速度。',
        'RMS诊断按活动RHS查询加权，包括自适应求解器拒绝的试探步，不是只统计接受轨迹。',
        '零读出、原生线性层捕获、patch往返、普通IG首批latent复现和零gamma退化均做真实网络检查。',
        '数据、模型、源码、两个新读出、噪声标签、全部125个batch及ADM评价身份已核验。',
        'FID/sFID用相同缓存特征上的独立FP64公式复算；不是独立特征提取或独立数据确认。', '',
        '首次训练在第一个原模型调用的embedding处因int16标签退出；显式转换为int64后重新冻结训练请求。',
        '修复前没有采集缓存、优化器更新或生成质量结果；原请求、源码和失败原因保存在amendments/01_label_dtype。', '',
        '本轮只做一次缓存特征读出更新，没有证明计算固定点存在、收敛或具有质量含义。',
        '已有探索噪声bank反复使用，当前结果不能充当独立确认。完整研究目标未达成。', '',
        f'生成请求SHA256：`{request_hash}`；原始资产：`{root}`。', '',
        '[冻结协议](SMALL_SIT_FEEDBACK_READER_PROTOCOL_20260909_ZH.md) · '
        '[指标](data/small_sit_feedback_reader_20260909/metrics.csv) · '
        '[审计](data/small_sit_feedback_reader_20260909/audit.json)', '',
        '固定首8张，未按质量挑选：', '', '![配对生成](data/small_sit_feedback_reader_20260909/paired_first8.png)']
    if cost_audit is not None:
        ratio = cost_audit['ratios_to_native']
        paragraph = ['所有采样进程退出后，另做固定状态的隔离计时，分别移除RMS诊断、保留真实读出计算。',
            f"Strong-only与Strong+gap纯读出RHS的耗时比为{ratio['strong_only_pure']:.4f}x、{ratio['strong_gap_pure']:.4f}x。",
            '6轮循环调整计时顺序，每轮每方法50次；纯读出与带诊断实现在该输入上逐元素相同。',
            '这约10%的固定RHS开销不等于全轨迹重新实测的成本；上表的实际生成成本仍包含原诊断。',
            '[隔离成本记录](data/small_sit_feedback_reader_20260909/cost_audit.json)。', '']
        where = lines.index('RMS诊断按活动RHS查询加权，包括自适应求解器拒绝的试探步，不是只统计接受轨迹。')
        lines[where:where] = paragraph
    REPORT.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=status['phase'], completed=len(rows)-1, report=str(REPORT))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    main(parser.parse_args().require_complete)
