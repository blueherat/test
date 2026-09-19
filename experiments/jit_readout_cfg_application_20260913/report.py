import argparse
import csv
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from . import core as m
from experiments.jit_readout_transfer_20260913 import common as c

OUT = c.WORK / 'docs/data/jit_readout_cfg_application_20260913'
REPORT = c.WORK / 'docs/JIT_READOUT_CFG_APPLICATION_RESULTS_20260913_ZH.md'
NAMES = dict(cfg_reference='Official CFG', cfg_more='Stronger CFG control', cfg_native='CFG + original IG', cfg_mlp='CFG + MLP IG')


def manifest():
    c.atomic(OUT / 'artifact_manifest.json', dict(report_sha256=c.sha(REPORT),
        generator_sha256=c.sha(Path(__file__)), files={str(p.relative_to(c.WORK)): c.sha(p)
        for p in OUT.iterdir() if p.name != 'artifact_manifest.json'}))


def build():
    assert c.read(m.ROOT / 'status.json')['phase'] == 'complete'
    c.verify(m.ROOT / m.STAGE / 'request.json')
    rows = c.read(m.ROOT / m.STAGE / 'results.json')
    by = {r['arm']: r for r in rows}
    assert set(by) == set(m.ARMS)
    decision = c.read(m.ROOT / 'decision.json')
    audit = c.read(m.ROOT / m.STAGE / 'audit.json')
    benchmark = c.read(m.ROOT / 'inference_benchmark.json')
    assert audit['passed'] and benchmark['complete']
    labels = np.load(m.ROOT / m.STAGE / 'inputs/labels.npy')
    np.testing.assert_array_equal(np.bincount(labels, minlength=1000), np.ones(1000, dtype=np.int64))
    for arm in m.ARMS:
        assert by[arm]['primary_samples'] == m.N and by[arm]['generated_paths'] == m.N
        assert by[arm]['samples_sha256'] == c.sha(m.ROOT / m.STAGE / arm / 'samples.npz')
        assert by[arm]['full_calls_per_output'] == 198 and by[arm]['prefix_calls_at_inference'] == 0
    quality = pd.DataFrame([{k: by[a][k] for k in ('arm', 'fid', 'inception_score', 'primary_samples',
        'generated_paths', 'full_calls_per_output', 'prefix_calls_at_inference', 'head_calls_per_output', 'seconds')}
        for a in m.ARMS])
    quality['delta_from_cfg'] = quality.fid - by['cfg_reference']['fid']
    timing = pd.DataFrame([dict(arm=a, median_seconds=benchmark['medians'][a],
        min_seconds=min(benchmark['seconds'][a]), max_seconds=max(benchmark['seconds'][a]),
        relative_to_cfg=benchmark['medians'][a] / benchmark['medians']['cfg_reference'] - 1,
        batch=benchmark['batch'], repeats=benchmark['repeats']) for a in m.ARMS])
    files = [m.PROTOCOL, c.TRAIN / 'summary.json', m.ROOT / 'inference_benchmark.json',
        m.ROOT / 'status.json', m.ROOT / 'decision.json', m.ROOT / m.STAGE / 'request.json',
        m.ROOT / m.STAGE / 'results.json', m.ROOT / m.STAGE / 'metrics.json', m.ROOT / m.STAGE / 'audit.json']
    files += [m.ROOT / m.STAGE / f'checks{i}.json' for i in range(3)]
    tables = dict(quality=quality, decision=pd.DataFrame([decision]), timing=timing,
        timing_repeats=pd.DataFrame([dict(arm=a, repeat=i + 1, seconds=v)
            for a, values in benchmark['seconds'].items() for i, v in enumerate(values)]),
        source_files=pd.DataFrame([dict(path=str(p), sha256=c.sha(p)) for p in files]),
        Sources=pd.DataFrame([
            dict(title='Frozen application protocol', source=str(m.PROTOCOL), role='Four fixed arms, gate, window and cost'),
            dict(title='JiT official implementation', source='https://github.com/LTH14/JiT', role='CFG and Heun50 trajectory'),
            dict(title='IG official implementation', source='https://github.com/CVL-UESTC/Internal-Guidance', role='Prior internal guidance and combination context'),
            dict(title='SGG', source='https://arxiv.org/html/2603.20584v1', role='Prior conditional and weak-model guidance'),
            dict(title='Independent JiT 5K confirmation', source=str(c.WORK / 'docs/JIT_READOUT_CONFIRM_RESULTS_20260913_ZH.md'), role='Entry gate; not pooled with this new 1K'),
            dict(title='Raw artifacts', source=str(m.ROOT), role='4000 new quality images and complete audit trail')]))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT / (name + '.csv'), index=False)
    with pd.ExcelWriter(OUT / 'source_data.xlsx', engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = min(70, max(16, max(len(str(x.value or '')) for x in col) + 2))
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    delta = quality.delta_from_cfg
    ax.barh(range(4), delta, color=['#999999', '#999999', '#999999', '#176b8d'])
    ax.set_yticks(range(4), [NAMES[a] for a in m.ARMS])
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=.8)
    ax.set_title('JiT-B/16: new paired 1K, official CFG with retained IG heads')
    ax.set_xlabel('FID difference from official CFG; lower is better')
    lo, hi = min(float(delta.min()), 0) - 2, max(float(delta.max()), 0) + 4
    ax.set_xlim(lo, hi)
    for i, r in enumerate(quality.itertuples()):
        ax.text(hi - .02 * (hi - lo), i, f'FID {r.fid:.3f}; full 198', ha='right', va='center')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / ('quality_comparison.' + ext), dpi=180)
    plt.close(fig)
    canvas = Image.new('RGB', (195 + 4 * 192, 65 + 4 * 210), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    draw.text((10, 8), 'JiT-B/16: first four samples in order; all four arms, no selection', font=font, fill='black')
    for j, label in enumerate(labels[:4]):
        draw.text((195 + j * 192, 34), f'index {j}; class {label}', font=font, fill='black')
    for i, arm in enumerate(m.ARMS):
        with np.load(m.ROOT / m.STAGE / arm / 'samples.npz') as data:
            pixels = data['arr_0'][:4]
        draw.text((10, 90 + i * 210), NAMES[arm], font=font, fill='black')
        draw.text((10, 115 + i * 210), f"FID {by[arm]['fid']:.2f}", font=font, fill='black')
        for j, pixel in enumerate(pixels):
            canvas.paste(Image.fromarray(pixel).resize((192, 192)), (195 + j * 192, 65 + i * 210))
    canvas.save(OUT / 'first4.png')
    passed = decision['passes_application_gate']
    opening = ('固定MLP组合通过本轮1K应用门槛，值得另用独立5K确认；目前还不是已确认的组合收益。'
        if passed else '固定MLP组合没有通过本轮1K应用门槛，停止该组合，不追加系数或窗口搜索。')
    rel = 'data/' + OUT.name + '/'
    text = ['# JiT读出在官方CFG设置中的应用结果\n\n', opening + '\n\n',
        quality[['arm', 'fid', 'inception_score', 'full_calls_per_output', 'prefix_calls_at_inference', 'head_calls_per_output']].to_markdown(index=False) + '\n\n',
        f"MLP组合相对单独CFG的FID差为{decision['mlp_minus_cfg']:.4f}，相对原IG组合为"
        f"{decision['mlp_minus_native_combination']:.4f}，相对增强CFG为{decision['mlp_minus_stronger_cfg']:.4f}。预设门槛要求三者都≤−1，且IS≥单独CFG的90%。"
        '它是继续/停止规则，不是统计显著性检验。\n\n',
        f'![质量比较]({rel}quality_comparison.png)\n\n',
        '四臂各1000张，每类1张，共4000张新质量样本，numpy seed2026121401。初始噪声、标签、权重、'
        'Heun50末步Euler、CFG=3和区间(.1,1)全部配对。未合并之前的1K或5K样本，也没有新增训练。'
        '两个组合均在既有CFG场上加.3*(strong−weak)，使用原depth4；前25个Heun完整步的两个RHS都启用IG，'
        '之后关闭。增强CFG控制在相同窗口把weak换成已经计算的unconditional，使每个RHS的strong总系数与组合相同；'
        '它包括基础CFG未开启的早期，不能简称为全程CFG=3.3。MLP采用保留3000步EMA，原弱头采用既有50K EMA。\n\n',
        '四个采样器均为每图198次完整前向；组合的50次小头读出发生在已计算的条件主干第4层。'
        '没有独立弱前缀、附加条件主干或第二条生成路径。MLP相对单独CFG增加整个1,777,152参数读出，'
        '仅相对原IG组合是替换净增4,608参数。\n\n',
        '同卡batch4，轮换预热后各三次完整采样及像素量化，所有臂保留相同调用计数hook。'
        '计时重复数少，不以完整前向次数相同宣称延迟严格相等。\n\n', timing.to_markdown(index=False) + '\n\n',
        '官方CFG完整轨迹与冻结原实现一致；两个组合和增强CFG在附加项为0时逐项退回CFG；'
        '三条非零修改轨迹均与独立参考实现逐项一致，组合的独立实现调用原features接口。'
        '所有图像、像素前状态、输入、标签、请求SHA、完整及各12层block调用、头调用均核验。'
        f"同缓存特征FP64复算FID最大绝对差{audit['max_fid_error']:.6g}，不属于独立特征提取验证。"
        '评价沿用nanogen imagenet_256_fid_stats，不与论文50K或不同参考统计的数值比较。\n\n',
        f'![固定首四张]({rel}first4.png)\n\n',
        '本实验检验已有读出在更强实际配置中的应用价值。'
        '[IG](https://github.com/CVL-UESTC/Internal-Guidance)及[SGG](https://arxiv.org/html/2603.20584v1)已有相关引导先例，'
        '不将相加形式当成核心创新；这也不是纯CFG的新改进。单个1K不能证明统计显著性或普适组合规律。\n\n',
        f'[冻结协议]({m.PROTOCOL.name}) · [独立5K的IG确认](JIT_READOUT_CONFIRM_RESULTS_20260913_ZH.md) · '
        f'[源数据工作簿]({rel}source_data.xlsx) · [核验记录]({rel}verification.json)。\n']
    REPORT.write_text(''.join(text))
    c.atomic(OUT / 'verification.json', dict(passed=True, samples=4000, arms=4, labels_balanced=True,
        per_class_per_arm=1, source_files={str(p): c.sha(p) for p in files}, audit=audit,
        visual_inspection_pending=True, independent_feature_extraction=False, goal_complete=False))
    manifest()
    print('Built', REPORT, decision, flush=True)


def review():
    saved = c.read(OUT / 'artifact_manifest.json')
    assert c.sha(REPORT) == saved['report_sha256'] and c.sha(Path(__file__)) == saved['generator_sha256']
    for path, digest in saved['files'].items():
        assert c.sha(c.WORK / path) == digest
    wb = openpyxl.load_workbook(OUT / 'source_data.xlsx', read_only=True, data_only=True)
    reconciled = []
    for sheet in wb.worksheets:
        with (OUT / (sheet.title + '.csv')).open() as stream:
            expected = list(csv.reader(stream))
        actual = list(sheet.values)
        assert len(actual) == len(expected)
        for row, wanted in zip(actual, expected):
            assert len(row) == len(wanted)
            for x, y in zip(row, wanted):
                if x is None:
                    assert y == ''
                elif isinstance(x, bool):
                    assert str(x) == y
                elif isinstance(x, (int, float)):
                    assert math.isclose(x, float(y), rel_tol=1e-12, abs_tol=1e-10)
                else:
                    assert str(x) == y
        reconciled.append(sheet.title)
    wb.close()
    links = []
    for link in re.findall(r'\]\(([^)]+)\)', REPORT.read_text()):
        if not link.startswith(('http://', 'https://', '#')):
            assert (REPORT.parent / link).exists()
            links.append(link)
    v = c.read(OUT / 'verification.json')
    for path, digest in v['source_files'].items():
        assert c.sha(path) == digest
    v.update(visual_inspection_pending=False, workbook_tables_reconciled=reconciled, local_links_verified=links,
        inspected_figures={p.name: c.sha(p) for p in OUT.glob('*.png')},
        visual_notes='Displayed both PNG figures; checked readable labels, values and fixed sample order.')
    c.atomic(OUT / 'verification.json', v)
    manifest()
    print('Final report verification passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visuals-reviewed', action='store_true')
    args = parser.parse_args()
    review() if args.visuals_reviewed else build()
