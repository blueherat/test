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

OUT = c.WORK / 'docs/data/jit_readout_confirm_20260913'
REPORT = c.WORK / 'docs/JIT_READOUT_CONFIRM_RESULTS_20260913_ZH.md'
NAMES = {'mlp': 'MLP IG', 'native_base': 'Original IG', 'adg': 'ADG', 'cfg_heun25': 'CFG, Heun25'}


def load():
    assert c.read(m.ROOT / 'status.json')['phase'] == 'complete'
    c.verify(m.ROOT / m.STAGE / 'request.json')
    c.verify(c.TRAIN / 'request.json')
    rows = c.read(m.ROOT / m.STAGE / 'results.json')
    by = {r['arm']: r for r in rows}
    assert set(by) == set(m.ARMS)
    audit = c.read(m.ROOT / m.STAGE / 'audit.json')
    benchmark = c.read(m.ROOT / 'inference_benchmark.json')
    assert audit['passed'] and benchmark['complete']
    api = c.read(m.ROOT / 'replacement_api_verification.json')
    assert api['passed'] and api['samples_replayed'] == 8 and api['new_quality_samples'] == 0
    for path, digest in api['sources'].items():
        assert c.sha(path) == digest
    for arm in m.ARMS:
        assert by[arm]['primary_samples'] == m.N and by[arm]['generated_paths'] == m.N
        assert by[arm]['samples_sha256'] == c.sha(m.ROOT / m.STAGE / arm / 'samples.npz')
        assert by[arm]['prefix_calls_at_inference'] == 0
    labels = np.load(m.ROOT / m.STAGE / 'inputs/labels.npy')
    np.testing.assert_array_equal(np.bincount(labels, minlength=1000), np.full(1000, 5))
    return by, audit, benchmark, c.read(m.ROOT / 'decision.json'), c.read(c.TRAIN / 'summary.json')


def figures(quality, by):
    fig, ax = plt.subplots(figsize=(9, 4.3), layout='constrained')
    delta = quality.delta_from_native
    ax.barh(range(len(quality)), delta,
        color=['#176b8d' if a == 'mlp' else '#bd862e' if a == 'cfg_heun25' else '#999999' for a in m.ARMS])
    ax.set_yticks(range(len(quality)), [NAMES[a] for a in m.ARMS])
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=.8)
    ax.set_title('JiT-B/16: independent paired 5K, 98–100 full forwards per image')
    ax.set_xlabel('FID difference from original IG; lower is better')
    lo, hi = min(float(delta.min()), 0) - 2, max(float(delta.max()), 0) + 8
    ax.set_xlim(lo, hi)
    for i, r in enumerate(quality.itertuples()):
        ax.text(hi - .02 * (hi - lo), i, f'FID {r.fid:.3f}; full {r.full_calls_per_output}', ha='right', va='center')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / ('quality_comparison.' + ext), dpi=180)
    plt.close(fig)
    canvas = Image.new('RGB', (185 + 4 * 192, 65 + 210 * len(m.ARMS)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    labels = np.load(m.ROOT / m.STAGE / 'inputs/labels.npy')
    draw.text((10, 8), 'JiT-B/16: first four samples in order; all arms, no selection', font=font, fill='black')
    for j, label in enumerate(labels[:4]):
        draw.text((185 + j * 192, 34), f'index {j}; class {label}', font=font, fill='black')
    for i, arm in enumerate(m.ARMS):
        with np.load(m.ROOT / m.STAGE / arm / 'samples.npz') as data:
            pixels = data['arr_0'][:4]
        draw.text((10, 90 + i * 210), NAMES[arm], font=font, fill='black')
        draw.text((10, 115 + i * 210), f"FID {by[arm]['fid']:.2f}", font=font, fill='black')
        for j, pixel in enumerate(pixels):
            canvas.paste(Image.fromarray(pixel).resize((192, 192)), (185 + j * 192, 65 + i * 210))
    canvas.save(OUT / 'first4.png')


def manifest():
    c.atomic(OUT / 'artifact_manifest.json', dict(report_sha256=c.sha(REPORT),
        generator_sha256=c.sha(Path(__file__)), files={str(p.relative_to(c.WORK)): c.sha(p)
        for p in OUT.iterdir() if p.name != 'artifact_manifest.json'}))


def build():
    by, audit, benchmark, decision, training = load()
    quality = pd.DataFrame([{k: by[a][k] for k in ('arm', 'fid', 'inception_score', 'primary_samples',
        'generated_paths', 'full_calls_per_output', 'prefix_calls_at_inference', 'head_calls_per_output', 'seconds')}
        for a in m.ARMS])
    quality['delta_from_native'] = quality.fid - by['native_base']['fid']
    timing = pd.DataFrame([dict(arm=a, median_seconds=benchmark['medians'][a],
        min_seconds=min(benchmark['seconds'][a]), max_seconds=max(benchmark['seconds'][a]),
        relative_to_native=benchmark['medians'][a] / benchmark['medians']['native_base'] - 1,
        batch=benchmark['batch'], repeats=benchmark['repeats']) for a in m.ARMS])
    files = [m.PROTOCOL, c.PROTOCOL, c.TRAIN / 'request.json', c.TRAIN / 'summary.json',
        m.ROOT / 'inference_benchmark.json', m.ROOT / 'decision.json', m.ROOT / 'status.json',
        m.ROOT / 'replacement_api_verification.json',
        m.ROOT / m.STAGE / 'request.json', m.ROOT / m.STAGE / 'metrics.json',
        m.ROOT / m.STAGE / 'results.json', m.ROOT / m.STAGE / 'audit.json']
    files += [m.ROOT / m.STAGE / f'checks{i}.json' for i in range(3)]
    tables = dict(quality=quality, decision=pd.DataFrame([decision]), timing=timing,
        timing_repeats=pd.DataFrame([dict(arm=a, repeat=i + 1, seconds=v)
            for a, values in benchmark['seconds'].items() for i, v in enumerate(values)]),
        source_files=pd.DataFrame([dict(path=str(p), sha256=c.sha(p)) for p in files]),
        Sources=pd.DataFrame([
            dict(title='Frozen 5K protocol', source=str(m.PROTOCOL), role='Fixed design, samples and gate'),
            dict(title='JiT 1K result and equal-training control', source=str(c.WORK / 'docs/JIT_READOUT_TRANSFER_RESULTS_20260913_ZH.md'), role='Previous independent screen; not pooled into 5K'),
            dict(title='JiT official implementation', source='https://github.com/LTH14/JiT', role='Official CFG trajectory comparison'),
            dict(title='JiT paper', source='https://arxiv.org/html/2511.13720v1', role='Model and native clean prediction'),
            dict(title='SSG', source='https://arxiv.org/html/2607.29122v1', role='Frozen intermediate adapters are prior work'),
            dict(title='Raw artifacts', source=str(m.ROOT), role='All 20000 images, input/state/count records and cached features')]))
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
    figures(quality, by)
    write_report(quality, timing, by, audit, decision, training)
    c.atomic(OUT / 'verification.json', dict(passed=True, samples=m.N * len(m.ARMS), arms=len(m.ARMS),
        labels_balanced=True, per_class_per_arm=5, independent_of_1k_bank=True,
        audit=audit, source_files={str(p): c.sha(p) for p in files},
        visual_inspection_pending=True, independent_feature_extraction=False, goal_complete=False))
    manifest()
    print('Built', REPORT, decision, flush=True)


def write_report(quality, timing, by, audit, decision, training):
    rel = 'data/' + OUT.name + '/'
    net = training['parameters']['mlp'] - training['native_parameters']
    passed = decision['passes_ig_5k_gate']
    opening = ('MLP读出在JiT-B/16的全新5K中保持了相对原IG及ADG的收益，通过预设确认门槛。' if passed
               else 'MLP读出没有在JiT-B/16的全新5K中通过预设确认门槛，不能把此前1K作为已确认收益。')
    practical = ('MLP的FID也低于接近同查询预算的CFG25；这仍是单组配对样本，需保留求解器差异及实际耗时。'
                 if decision['mlp_fid_better_than_cfg25'] else
                 '接近同查询预算的CFG25仍有更低FID。因此，读出对IG的改善不能写成已优于CFG的实用方法。')
    paragraphs = [
        '# JiT-B/16中间读出：独立5K确认与CFG预算对照\n\n', opening + practical + '\n\n',
        quality[['arm', 'fid', 'inception_score', 'primary_samples', 'full_calls_per_output', 'prefix_calls_at_inference']].to_markdown(index=False) + '\n\n',
        f"MLP相对原IG的FID差为{decision['mlp_minus_native']:.4f}，相对ADG为{decision['mlp_minus_adg']:.4f}，"
        f"相对CFG25为{decision['mlp_minus_cfg25']:.4f}。门槛要求前两者均≤−1，且MLP IS≥原IG的90%；"
        '它是预设继续/停止规则，不是统计显著性检验。\n\n',
        f'![5K质量比较]({rel}quality_comparison.png)\n\n',
        '本轮四臂各5000张，共20000张；ImageNet-1K每类5张，numpy seed2026121391，batch4。'
        '四臂完全配对初始噪声和标签；原1K未合并进入本轮FID。没有重新训练或挑选checkpoint，'
        'MLP仍为原3000步最终EMA，depth4、alpha=.3、Euler100、前50步启用保持原样。'
        'ADG保持原弱头、强度和窗口。\n\n',
        'CFG使用官方CFG=3、区间(.1,1)、Heun25且末步Euler。49次RHS各计算条件和无条件输出，'
        '合计98次主干前向；三个IG相关臂100次主干前向。CFG25与官方Denoiser两条完整轨迹逐项相同，'
        '三个IG相关臂直接调用冻结1K函数，并通过原强/弱输出、旧IG轨迹及零引导一致性预检。'
        'CFG与IG求解器和时间网格不同，这是一项接近同查询预算的实用比较，不能用于单独识别求解器或引导公式的因果贡献。\n\n',
        '同GPU、batch4，轮换预热后各三次完整采样和像素量化；所有臂保留相同主干与block调用计数hook。'
        '仅三次计时不能支持精确的加速幅度或延迟置信区间。\n\n', timing.to_markdown(index=False) + '\n\n',
        f"MLP弱头{training['parameters']['mlp']:,}参数，原弱头{training['native_parameters']:,}参数，替换净增{net:,}。"
        '只在原主干第4层输出后计算选中的小头，每图额外prefix为0；比较进程为了切换方法加载多个小头，'
        '上述参数差针对部署时仅保留一个弱头的替换，不是进程总显存差。'
        '相对无IG的裸主干，MLP整个小头都是额外参数。\n\n',
        '另提供只加载一个主干和一个MLP弱头的[采样入口](../experiments/jit_readout_confirm_20260913/reference.py)。'
        '它重放本轮前8张，最终FP32状态及uint8像素逐项一致；零引导退回条件模型也逐项一致。'
        '这些8张是重放核验，不计入新质量样本。运行前设置所需CUDA_VISIBLE_DEVICES；'
        'noise为CUDA上的FP32张量[N,3,256,256]，labels为CUDA上的int64类别索引[N]，范围0–999。\n\n'
        '```python\nfrom experiments.jit_readout_confirm_20260913.reference import Reference\n'
        'rt = Reference()\nstates, counts = rt.sample(noise, labels)\nimages = rt.pixels(states)\nrt.close()\n```\n\n',
        '等训练原结构对照已在前一轮独立1K完成，本5K未重复该臂。因此，这里验证的是固定MLP对既有IG/ADG的迁移收益，'
        '不将其写成5K的等训练结构因果实验，也不把读出参数化、标准化及位置输入的联合变化全部归因于非线性。\n\n',
        f"全部图像、像素前状态、量化、输入噪声、标签、源码/资产SHA和每批12层调用数核验通过。"
        f"同一缓存特征用FP64对称协方差形式复算FID，最大绝对差{audit['max_fid_error']:.6g}。"
        '沿用同一nanogen imagenet_256_fid_stats；未换成JiT仓库统计，'
        '不能将本5K绝对FID与论文50K数值比较，复算也不等于独立特征提取器验证。\n\n',
        f'![固定前四张]({rel}first4.png)\n\n',
        '展示采样顺序的前四张及相同类别，未按观感选择。该结果不能证明共同误差分布、平滑或加性误差抵消，'
        '也不能仅凭冻结小读出宣称新的核心原理。'
        '[JiT官方实现](https://github.com/LTH14/JiT)是模型与CFG来源；'
        '[SSG](https://arxiv.org/html/2607.29122v1)已有冻结中间adapter的先例。\n\n',
        f'[冻结5K协议]({m.PROTOCOL.name}) · [前轮1K及等训练控制](JIT_READOUT_TRANSFER_RESULTS_20260913_ZH.md) · '
        f'[源数据工作簿]({rel}source_data.xlsx) · [核验记录]({rel}verification.json)。\n'
    ]
    REPORT.write_text(''.join(paragraphs))


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
