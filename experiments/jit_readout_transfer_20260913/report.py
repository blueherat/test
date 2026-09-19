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
from . import common as c

OUT = c.WORK / 'docs/data/jit_readout_transfer_20260913'
REPORT = c.WORK / 'docs/JIT_READOUT_TRANSFER_RESULTS_20260913_ZH.md'


def build():
    assert c.read(c.ROOT / 'status.json')['phase'] == 'complete'
    c.verify(c.ROOT / c.STAGE / 'request.json')
    c.verify(c.TRAIN / 'request.json')
    rows = c.read(c.ROOT / c.STAGE / 'results.json')
    decision = c.read(c.ROOT / 'decision.json')
    audit = c.read(c.ROOT / c.STAGE / 'audit.json')
    benchmark = c.read(c.ROOT / 'inference_benchmark.json')
    training = c.read(c.TRAIN / 'summary.json')
    assert audit['passed'] and benchmark['complete']
    by = {r['arm']: r for r in rows}
    assert set(by) == set(c.ARMS)
    quality = pd.DataFrame([{k: by[a][k] for k in ('arm', 'fid', 'inception_score', 'primary_samples',
        'generated_paths', 'full_calls_per_output', 'prefix_calls_at_inference', 'head_calls_per_output', 'seconds')} for a in c.ARMS])
    quality['delta_from_native'] = quality.fid - by['native_base']['fid']
    timing = pd.DataFrame([dict(arm=a, median_seconds=v, min_seconds=min(benchmark['seconds'][a]),
        max_seconds=max(benchmark['seconds'][a]), relative_to_native=v / benchmark['medians']['native_base'] - 1,
        batch=benchmark['batch'], repeats=benchmark['repeats']) for a, v in benchmark['medians'].items()])
    training_table = pd.DataFrame([dict(arm=a, validation_mse=v,
        parameters=training['parameters'].get(a, training['native_parameters']))
        for a, v in training['validation_mse'].items()])
    files = [c.PROTOCOL, c.TRAIN / 'request.json', c.TRAIN / 'summary.json', c.ROOT / 'implementation_check.json',
        c.ROOT / 'inference_benchmark.json', c.ROOT / 'decision.json', c.ROOT / 'status.json',
        c.ROOT / c.STAGE / 'request.json', c.ROOT / c.STAGE / 'metrics.json', c.ROOT / c.STAGE / 'results.json',
        c.ROOT / c.STAGE / 'audit.json']
    tables = dict(quality=quality, decision=pd.DataFrame([decision]), timing=timing,
        timing_repeats=pd.DataFrame([dict(arm=a, repeat=i + 1, seconds=v)
            for a, values in benchmark['seconds'].items() for i, v in enumerate(values)]),
        training=training_table, training_history=pd.DataFrame(training['history']),
        source_files=pd.DataFrame([dict(path=str(p), sha256=c.sha(p)) for p in files]),
        Sources=pd.DataFrame([
            dict(title='Frozen protocol', source=str(c.PROTOCOL), role='Fixed design and gate'),
            dict(title='JiT official implementation', source='https://github.com/LTH14/JiT', role='Model, native prediction and CFG stepper'),
            dict(title='JiT paper', source='https://arxiv.org/html/2511.13720v1', role='Pixel-space clean prediction'),
            dict(title='ADG', source='https://arxiv.org/html/2506.11039v1', role='Existing angular guidance control'),
            dict(title='SSG', source='https://arxiv.org/html/2607.29122v1', role='Prior frozen intermediate adapters; no synthetic training used here'),
            dict(title='Raw artifacts', source=str(c.ROOT), role='Images, states, labels, hashes and cached features')]))
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
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    delta = quality.delta_from_native
    ax.barh(range(len(quality)), delta,
        color=['#176b8d' if a == 'mlp' else '#d2b36b' if a == 'cfg_reference' else '#999999' for a in c.ARMS])
    ax.set_yticks(range(len(quality)), c.ARMS)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=.8)
    ax.set_title('JiT-B/16 ImageNet-1K: paired 1K transfer screen')
    ax.set_xlabel('FID difference from original IG; lower is better')
    lo, hi = min(float(delta.min()), 0) - 2, max(float(delta.max()), 0) + 8
    ax.set_xlim(lo, hi)
    for i, r in enumerate(quality.itertuples()):
        ax.text(hi - .02 * (hi - lo), i, f'FID {r.fid:.3f}; full {r.full_calls_per_output}', ha='right', va='center')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / ('quality_comparison.' + ext), dpi=180)
    plt.close(fig)
    canvas = Image.new('RGB', (185 + 4 * 192, 45 + 210 * len(c.ARMS)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    draw.text((10, 10), 'JiT-B/16: first four samples in order, all six arms; no selection', font=font, fill='black')
    for i, arm in enumerate(c.ARMS):
        with np.load(c.ROOT / c.STAGE / arm / 'samples.npz') as data:
            pixels = data['arr_0'][:4]
        draw.text((10, 70 + i * 210), arm, font=font, fill='black')
        draw.text((10, 95 + i * 210), f"FID {by[arm]['fid']:.2f}", font=font, fill='black')
        for j, pixel in enumerate(pixels):
            canvas.paste(Image.fromarray(pixel).resize((192, 192)), (185 + j * 192, 45 + i * 210))
    canvas.save(OUT / 'first4.png')
    passed = decision['passes_transfer_gate']
    intro = ('MLP读出在JiT-B/16的这组固定1K通过迁移门槛，值得用全新5K确认。'
             if passed else 'SiT有效的MLP读出没有在这次JiT-B/16固定1K通过迁移门槛，停止这次构造，不补扫参数。')
    intro += (f"MLP FID为{by['mlp']['fid']:.4f}，旧IG为{by['native_base']['fid']:.4f}，"
              f"ADG为{by['adg']['fid']:.4f}，等训练原结构为{by['native_fresh']['fid']:.4f}。")
    rel = 'data/' + OUT.name + '/'
    net = training['parameters']['mlp'] - training['native_parameters']
    text = [f'# JiT-B/16中间MLP读出的实际生成检验\n\n{intro}\n\n',
        '六臂各1000张新图，共6000张，ImageNet-1K每类1张；numpy seed2026121381，初始像素噪声和标签全部配对。'
        'MLP、等训练原结构和旧IG沿用depth4、Euler100、alpha=.3、前50步启用、后50步关闭；'
        'ADG使用旧弱头及同一强度和窗口。没有叠加CFG，没有重选层数、窗口或系数。\n\n',
        quality[['arm', 'fid', 'inception_score', 'full_calls_per_output', 'prefix_calls_at_inference']].to_markdown(index=False) + '\n\n',
        f'![质量比较]({rel}quality_comparison.png)\n\n',
        '官方CFG参考为CFG=3、Heun50末步Euler、区间(.1,1)、198次full；其余为100次full。'
        'CFG是更多主干计算的实用参考，不以其差异识别读出结构效果，也不将这里的1K FID与论文50K数值横向比较。\n\n',
        f"实际替换参数：旧弱头{training['native_parameters']:,}，MLP {training['parameters']['mlp']:,}，净增{net:,}。"
        '采样只在原模型第4层hook中计算选中的弱头，不重复前缀，也不同时计算原弱头。'
        '比较程序为切换方法同时加载三个小头；参数净差指部署时保留一个替换弱头的比较，不能当作比较进程总显存差。\n\n',
        '同GPU、batch4，轮换预热后各三次完整采样及像素量化；全部方法保留相同调用计数hook。'
        '三次计时不提供精确的延迟置信区间。\n\n', timing.to_markdown(index=False) + '\n\n',
        f"两个新头共享3000步真实图像训练，batch32；更新循环约{training['training_seconds']:.2f}秒，不含加载、"
        '统计批和最终验证。MLP沿用SiT有效结构，但输出改为JiT原生clean patch，采用原生logit-normal时间和velocity loss分母下限。'
        '每类48张，共48000张真实训练图，两次打乱遍历；训练外另有32批统计估计。'
        '主干与原条件末层权重及梯度隔离均核验。没有strong生成数据或蒸馏。\n\n',
        training_table.to_markdown(index=False) + '\n\n',
        '原结构对照使用官方RMSNorm权重1、输出及AdaLN末层为0的初始化。新头的优化预算相同，'
        '但原结构与MLP的标准化、位置输入及参数化不同；固定3000步不证明都已收敛。'
        '旧IG使用既有50K弱头，与新训练预算不同。预测MSE未用于选择checkpoint或判断生成收益。\n\n',
        '共享强输出、旧弱输出、旧IG完整轨迹、零引导退回条件模型、官方CFG完整轨迹均逐项一致。'
        '每个采样批都记录12层实际调用与完整调用数，全部为0额外prefix。'
        f"所有图像、状态量化、标签、输入及请求SHA核验通过；缓存特征FP64复算FID最大差{audit['max_fid_error']:.6g}。"
        '评价使用既有nanogen的imagenet_256_fid_stats，未换成JiT仓库自带统计；所有臂使用同一参考，'
        '不将复算称为独立特征提取器验证。\n\n',
        f'![固定前四张]({rel}first4.png)\n\n',
        '每类只有一张，这轮不支持类别内质量估计或多种子统计显著性。它检验固定SiT结构迁移到JiT的结果，'
        '不证明共同误差分布、平滑、加性误差抵消或普适跨模型规律。'
        '[JiT官方代码](https://github.com/LTH14/JiT)提供模型与CFG来源；'
        '[SSG](https://arxiv.org/html/2607.29122v1)已有冻结中间adapter先例，小MLP本身不构成核心新意。\n\n',
        f'[冻结协议]({c.PROTOCOL.name}) · [源数据工作簿]({rel}source_data.xlsx) · [核验记录]({rel}verification.json)。\n']
    REPORT.write_text(''.join(text))
    labels = np.load(c.ROOT / c.STAGE / 'inputs/labels.npy')
    np.testing.assert_array_equal(np.bincount(labels, minlength=1000), np.ones(1000, dtype=np.int64))
    verification = dict(passed=True, samples=6000, labels_balanced=True, audit=audit,
        source_files={str(p): c.sha(p) for p in files}, visual_inspection_pending=True,
        independent_feature_extraction=False, goal_complete=False)
    c.atomic(OUT / 'verification.json', verification)
    manifest()
    print('Report built', REPORT, decision, flush=True)


def manifest():
    c.atomic(OUT / 'artifact_manifest.json', dict(report_sha256=c.sha(REPORT),
        generator_sha256=c.sha(Path(__file__)), files={str(p.relative_to(c.WORK)): c.sha(p)
        for p in OUT.iterdir() if p.name != 'artifact_manifest.json'}))


def review():
    saved = c.read(OUT / 'artifact_manifest.json')
    assert c.sha(REPORT) == saved['report_sha256'] and c.sha(Path(__file__)) == saved['generator_sha256']
    for path, digest in saved['files'].items():
        assert c.sha(c.WORK / path) == digest
    wb = openpyxl.load_workbook(OUT / 'source_data.xlsx', read_only=True, data_only=True)
    tables = []
    for sheet in wb.worksheets:
        with (OUT / (sheet.title + '.csv')).open() as f:
            expected = list(csv.reader(f))
        actual = list(sheet.values)
        assert len(expected) == len(actual)
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
        tables.append(sheet.title)
    wb.close()
    links = []
    for link in re.findall(r'\]\(([^)]+)\)', REPORT.read_text()):
        if not link.startswith(('http://', 'https://', '#')):
            assert (REPORT.parent / link).exists()
            links.append(link)
    v = c.read(OUT / 'verification.json')
    for path, digest in v['source_files'].items():
        assert c.sha(path) == digest
    v.update(visual_inspection_pending=False, workbook_tables_reconciled=tables, local_links_verified=links,
        inspected_figures={p.name: c.sha(p) for p in OUT.glob('*.png')},
        visual_notes='Displayed all listed PNGs and checked labels and fixed sample order.')
    c.atomic(OUT / 'verification.json', v)
    manifest()
    print('Final report verification passed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--visuals-reviewed', action='store_true')
    a = p.parse_args()
    review() if a.visuals_reviewed else build()
