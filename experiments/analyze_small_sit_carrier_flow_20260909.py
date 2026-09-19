"""Audit completed carrier-flow controls and render an evidence-linked readout."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.analyze_official_sit_pfr_symmetric_20260909 import fid_components
from experiments.small_sit_carrier_flow_20260909 import (
    ARMS, BANK_ROOT, BATCH, RANKS, REFERENCE, ROOT, SAMPLES, WORK, verify_request,
)
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha

OUT = WORK / 'docs/data/small_sit_carrier_flow_20260909'
REPORT = WORK / 'docs/SMALL_SIT_CARRIER_FLOW_RESULTS_20260909_ZH.md'
TITLES = dict(ig_restarted='普通IG，同分段', write_before='先写入，再Strong', write_after='先Strong，再写入')


def font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def contact_sheet(results):
    indices = list(range(8))
    labels = np.load(BANK_ROOT / 'labels.npy')
    columns = len(results)
    side, margin, header, row_label = 192, 12, 62, 23
    canvas = Image.new('RGB', (columns * (side + margin) + margin,
                              header + len(indices) * (side + row_label + margin)), '#f8f9fc')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 7), 'Fixed first 8 samples; identical noise and labels', fill='#172033', font=font(14))
    for column, rec in enumerate(results):
        x = margin + column * (side + margin)
        draw.text((x, 31), rec['arm'], fill='#172033', font=font(14))
        with np.load(rec['sample_path']) as data:
            for row, index in enumerate(indices):
                y = header + row * (side + row_label + margin)
                draw.text((x, y), f'index {index}, class {labels[index]}', fill='#364359', font=font(13))
                canvas.paste(Image.fromarray(data['arr_0'][index]).resize((side, side), Image.Resampling.LANCZOS),
                             (x, y + row_label))
    path = OUT / 'paired_first8.png'
    canvas.save(path)
    return dict(indices=indices, selection='first eight input indices, fixed without quality selection',
                path=str(path), sha256=sha(path))


def main(require_complete):
    request, request_hash = verify_request()
    status = read(ROOT / 'status.json')
    results = read(ROOT / 'results.json') if (ROOT / 'results.json').exists() else []
    if require_complete:
        assert status['phase'] == 'complete' and len(results) == len(ARMS)
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(REFERENCE) as ref:
        mu, cov = ref['mu'].astype(float), ref['sigma'].astype(float)
        mu_s, cov_s = ref['mu_s'].astype(float), ref['sigma_s'].astype(float)
    checks, rows, endpoint = [], [], {}
    for rec in results:
        assert rec['request_sha256'] == request_hash and rec['arm'] in ARMS
        if not rec['complete']:
            continue
        arm = rec['arm']
        d = ROOT / arm
        assert rec['coverage_verified'] and rec['samples'] == SAMPLES
        assert sha(rec['sample_path']) == rec['sample_sha256']
        assert sha(d / 'latents.npy') == rec['latents_sha256']
        assert sha(d / 'activations.npz') == rec['activations_sha256']
        assert rec['noise_sha256'] == request['bank']['noise_sha256']
        assert rec['label_sha256'] == request['bank']['label_sha256']
        cache = OUT / f'audit_{arm}.json'
        key = dict(request_sha256=request_hash, activations_sha256=rec['activations_sha256'],
                   reference_sha256=request['reference_sha256'], fid=rec['fid'], sfid=rec['metrics']['sfid'])
        if cache.exists():
            check = read(cache)
            assert check['key'] == key
        else:
            with np.load(d / 'activations.npz') as acts:
                assert len(acts['pool_3']) == SAMPLES and len(acts['spatial']) == SAMPLES
                fm, fc = fid_components(acts['pool_3'].astype(float), mu, cov)
                sm, sc = fid_components(acts['spatial'].astype(float), mu_s, cov_s)
            check = dict(key=key, independent_fid=fm + fc, independent_sfid=sm + sc,
                         fid_absolute_error=abs(fm + fc - rec['fid']),
                         sfid_absolute_error=abs(sm + sc - rec['metrics']['sfid']),
                         independent_arithmetic_only=True, independent_feature_extraction=False)
            assert check['fid_absolute_error'] < .001 and check['sfid_absolute_error'] < .001, check
            atomic(cache, check)
        checks.append(dict(arm=arm, **check))
        block_nfe, writes, states = [], [], []
        for rank in range(RANKS):
            for batch in sorted((d / f'rank{rank}').glob('batch*.npz')):
                with np.load(batch) as b:
                    block_nfe.append(b['block_full_calls'])
                    writes.append(b['block_write_rms'])
                    states.append(b['block_state_rms'])
        assert len(block_nfe) == SAMPLES // BATCH
        rows.append(dict(arm=arm, fid=rec['fid'], sfid=rec['metrics']['sfid'],
            inception_score=rec['metrics']['inception_score'], full_calls_per_image=rec['full_calls_per_image'],
            auxiliary_full_calls_per_image=rec['auxiliary_full_calls_per_image'],
            trajectory_gpu_seconds=rec['trajectory_gpu_seconds'], decode_gpu_seconds=rec['decode_gpu_seconds'],
            sum_batch_gpu_seconds=rec['sum_batch_gpu_seconds'],
            mean_full_calls_by_block=np.mean(block_nfe, axis=0).tolist(),
            mean_write_rms_by_block=np.concatenate(writes, axis=1).mean(axis=1).tolist(),
            mean_state_rms_by_block=np.concatenate(states, axis=1).mean(axis=1).tolist()))
        endpoint[arm] = np.load(d / 'latents.npy').astype(float)
    paired = []
    for a, b in itertools.combinations(endpoint, 2):
        x, y = endpoint[a].reshape(SAMPLES, -1), endpoint[b].reshape(SAMPLES, -1)
        per_image = np.sqrt(np.mean((x - y) ** 2, axis=1))
        paired.append(dict(first=a, second=b, mean_endpoint_difference_rms=float(per_image.mean()),
            median_endpoint_difference_rms=float(np.median(per_image)),
            relative_rms_to_first=float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(x ** 2))),
            exact_latent_match=bool(np.array_equal(x, y))))
    image_record = contact_sheet([rec for rec in results if rec['complete']]) if rows else None
    audit = dict(status=status, request_sha256=request_hash, source_and_asset_hashes_passed=True,
                 completed_arms=len(rows), arithmetic_checks=checks, paired_endpoints=paired,
                 fixed_examples=image_record, rows=rows, research_goal_achieved=False)
    atomic(OUT / 'audit.json', audit)
    if rows:
        with (OUT / 'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# 小SiT：有限修正流的真实生成对照', '']
    if len(rows) == 3 and all(row['fid'] > rows[0]['fid'] for row in rows[1:]):
        lines += ['本轮两个有限写入方案的FID均落后于同分段普通IG，且均增加约18%的采样与解码成本。',
                  '先写入的sFID和IS略好，但没有FID优势；按协议结束这组三配置筛查，不自动扩大步数、幅度或horizon。', '']
    lines += [f"状态：{status['phase']}；完整质量结果 {len(rows)}/3。四卡预检通过后直接做真实1K采样，没有新增toy前置实验。", '',
        '固定ImageNet-100 SiT-S/2 v800K EMA、depth4 v50K；三组共用历史噪声/标签，',
        '主积分同为Dopri5与0/.125/.25/.375/.5/1分段。额外IG强度为前半段.6/.7、后半段0。',
        '具体操作与冻结范围见[协议](SMALL_SIT_CARRIER_FLOW_PROTOCOL_20260909_ZH.md)。', '',
        '| 配置 | FID↓ | sFID↓ | IS↑ | Full调用/图 | 总batch GPU秒 | 相对IG成本 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    base = next((row for row in rows if row['arm'] == 'ig_restarted'), None)
    for row in rows:
        ratio = row['sum_batch_gpu_seconds'] / base['sum_batch_gpu_seconds'] if base else None
        lines.append(f"| {TITLES[row['arm']]} | {row['fid']:.6f} | {row['sfid']:.6f} | {row['inception_score']:.6f} | "
                     f"{row['full_calls_per_image']:.3f} | {row['sum_batch_gpu_seconds']:.2f} | {ratio:.3f}x |")
    if len(rows) == 3:
        best = min(rows[1:], key=lambda row: row['fid'])
        gain = (base['fid'] - best['fid']) / base['fid'] * 100
        change = f'降低{gain:.3f}%' if gain >= 0 else f'升高{-gain:.3f}%'
        lines += ['', f"本轮两个写入方案中最低FID来自{TITLES[best['arm']]}；相对新普通IG的FID{change}。",
            '这只是已用于历史探索的配对1K结果；不能作为独立确认、最优强度比较或新颖性证据。']
    lines += ['', '历史未分段普通IG为64.851298，历史lifting为65.286964；分段网格不同，均单列为背景。',
        '旧普通IG完整像素已按历史规则删除，本轮没有声称旧像素复现。', '',
        '写入方案固定原时间，沿D=S−W做4个Heun子步；每个活动段重新读取两头8次，总额外32次Full/图。',
        '这属于Strong流与修正流的标准分裂；当前实验没有证明有限承载等式成立，也没有建立新的高质量固定点。',
        '先写或后写引起的终点变化属于描述性证据，不能替代FID等质量结果。', '',
        '| 终点对照 | 平均差值RMS | 相对首组RMS |', '|---|---:|---:|']
    for pair in paired:
        lines.append(f"| {TITLES[pair['first']]} / {TITLES[pair['second']]} | {pair['mean_endpoint_difference_rms']:.6f} | "
                     f"{pair['relative_rms_to_first']:.6f} |")
    lines += ['', '成本是所有batch采样与解码耗时之和，含各组相同的分段记录开销；不含模型加载、入口检查和ADM评价。',
        '每图调用数是各批NFE的平均值；自适应主积分调用数可能不同，两个写入方案只固定辅助调用预算。',
        '官方ADM特征上的FID与sFID经独立FP64样本空间计算核验（误差<.001）；这是算术核验，非独立数据或特征复测。', '',
        '冻结请求包含采样代码、直接评价入口、模型、VAE及reference身份；`evaluation_dependencies.json`另存ADM实现与Inception图的哈希，',
        '该补充身份记录发生在前两次评价之后，不冒称提前冻结的依赖快照。', '',
        f'冻结请求SHA256：`{request_hash}`。原始数据：`{ROOT}`。', '',
        '[指标CSV](data/small_sit_carrier_flow_20260909/metrics.csv) · [完整审计](data/small_sit_carrier_flow_20260909/audit.json)']
    if rows:
        lines += ['', '固定展示输入索引0–7，没有按结果挑图：', '',
                  '![相同输入的三组生成](data/small_sit_carrier_flow_20260909/paired_first8.png)']
    REPORT.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(completed=len(rows), report=str(REPORT), audit=str(OUT / 'audit.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    main(parser.parse_args().require_complete)
