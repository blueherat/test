"""Independent confirmation of a control-arm signal; no novelty claim."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import openpyxl
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import audit, mixture
from . import core as m

OUT = c.WORK / 'docs/data/context_reference_confirm_20260912'
REPORT = c.WORK / 'docs/CONTEXT_REFERENCE_CONFIRM_RESULTS_20260912_ZH.md'


def finalize():
    m.configure()
    mixture.ROOT = m.ROOT
    torch.set_num_threads(4)
    while c.read(m.ROOT / 'status.json')['phase'] != 'complete':
        state = c.read(m.ROOT / 'status.json')
        if not Path('/proc', str(state['pid'])).exists():
            raise RuntimeError('Confirmation controller exited before completion')
        time.sleep(5)
    m.verify('sit_small')
    results = c.read(m.ROOT / 'sit_small' / m.STAGE / 'results.json')
    assert len(results) == 6
    by = {r['arm']: r for r in results}
    audits = [audit.arm('sit_small', m.STAGE, r['arm']) for r in results]
    rows = [{k: r[k] for k in ('arm', 'fid', 'inception_score', 'seconds', 'primary_samples',
                              'generated_paths', 'full_calls_per_output', 'prefix_calls_at_inference')} for r in results]
    quality = pd.DataFrame(rows)
    candidate = by['context_base']
    controls = [r for r in results if r['arm'] != 'context_base']
    best = min(controls, key=lambda r: r['fid'])
    passed = candidate['fid'] <= best['fid'] - 1 and candidate['inception_score'] >= .9 * by['native_base']['inception_score']
    decision = dict(model='sit_small', candidate='context_base', candidate_fid=candidate['fid'],
        best_control=best['arm'], best_control_fid=best['fid'], delta=candidate['fid'] - best['fid'],
        native_delta=candidate['fid'] - by['native_base']['fid'], adg_delta=candidate['fid'] - by['adg']['fid'],
        inception_ratio=candidate['inception_score'] / by['native_base']['inception_score'],
        passes_1k_resource_gate=passed, original_candidate_was_a_control=True,
        independent_generation_bank=True, novelty_established=False, goal_complete=False)
    original = c.read(c.EXPS / 'input_local_completion_20260912/sit_small/screen_400/results.json')
    original = {r['arm']: r for r in original}
    selection = pd.DataFrame([dict(stage='exploratory_400', arm=arm, fid=original[arm]['fid'],
        inception_score=original[arm]['inception_score']) for arm in ('context_base', 'native_base', 'native_half', 'native_double', 'strong')])
    bench = c.read(c.EXPS / 'input_local_completion_20260912/sit_small/inference_benchmark.json')
    timing = pd.DataFrame([dict(batch=bench['batch'], repeats=bench['repeats'],
        native_seconds=bench['medians']['native_base'], context_seconds=bench['medians']['context_base'],
        relative_change=bench['relative_changes']['context_base'], extra_parameters=bench['extra_parameters_per_candidate'],
        provenance='same retained head and runtime; benchmark from preceding completion stage')])
    sources = pd.DataFrame([
        dict(title='Frozen independent confirmation protocol', location=str(m.PROTOCOL), role='Post-hoc selection and six-arm 1K test, no parameter changes'),
        dict(title='Raw source and sample artifacts', location=str(m.ROOT), role='1000 images per arm, identical input bank and source hashes'),
        dict(title='Original local-head test', location=str(c.EXPS / 'input_local_completion_20260912'), role='Complete selection data; not independent confirmation'),
        dict(title='ADG prior work', location='https://arxiv.org/html/2506.11039v1', role='Existing angular denoiser guidance; baseline implementation inherited'),
        dict(title='Synthetic Self-Guidance prior work', location='https://arxiv.org/html/2607.29122v1', role='Frozen-backbone intermediate-head principle already exists')])
    OUT.mkdir(parents=True, exist_ok=True)
    tables = dict(quality=quality, decision=pd.DataFrame([decision]), selection=selection, timing=timing)
    for name, df in tables.items():
        df.to_csv(OUT / (name + '.csv'), index=False)
    c.atomic(OUT / 'decision.json', decision)
    with pd.ExcelWriter(OUT / 'source_data.xlsx', engine='openpyxl') as writer:
        for name, df in dict(tables, Sources=sources).items():
            df.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = min(65, max(16, max(len(str(v.value or '')) for v in col) + 2))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    delta = quality.fid - by['native_base']['fid']
    ax.barh(range(6), delta, color=['#176b8d' if arm == 'context_base' else '#999999' for arm in quality.arm])
    ax.set_yticks(range(6), quality.arm)
    ax.invert_yaxis()
    ax.axvline(0, lw=.8, color='black')
    ax.set_xlabel('FID difference from native IG; lower is better')
    ax.set_title('SiT-S/2: independent 1K check of the contextual readout')
    lo, hi = min(-2, float(delta.min()) - 3), max(8, float(delta.max()) + 10)
    ax.set_xlim(lo, hi)
    for i, r in enumerate(quality.itertuples()):
        ax.text(hi - .02 * (hi - lo), i, f'FID {r.fid:.3f}', ha='right', va='center')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / ('quality_comparison.' + ext), dpi=180)
    plt.close(fig)
    canvas = Image.new('RGB', (185 + 192 * 4, 45 + 210 * 3), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    draw.text((10, 8), 'SiT-S/2: independent bank; first four in order; no selection', font=font, fill='black')
    for i, arm in enumerate(('native_base', 'context_base', 'adg')):
        root = m.ROOT / 'sit_small' / m.STAGE / arm
        with np.load(root / 'samples.npz') as data:
            pixels = data['arr_0'][:4]
        draw.text((10, 70 + i * 210), arm, font=font, fill='black')
        draw.text((10, 93 + i * 210), f"FID {by[arm]['fid']:.2f}", font=font, fill='black')
        for j, image in enumerate(pixels):
            canvas.paste(Image.fromarray(image).resize((192, 192)), (185 + j * 192, 45 + i * 210))
    canvas.save(OUT / 'first4.png')
    conclusion = ('上下文读出通过预先规定的1K资源门槛，仍需要新的5K与更严格成本比较，尚无核心方法成功结论。'
                  if passed else '上下文读出未通过预先规定的独立1K资源门槛，当前对照线索结束，不追加5K或参数搜索。')
    text = [
        '# 上下文读出对照的独立1K核查\n\n', conclusion + '\n\n',
        '上一阶段Local参考失败，但SiT的Context对照在400图上比原IG最好强度低2.97 FID。'
        '本轮明确将这个对照作为事后选择，另取1000个全新噪声、每类10图，与原IG三档幅度、ADG和Strong同时比较。'
        '未更新训练、checkpoint、窗口或强度；该1K只验证具体读出信号，不能证明冻结主干加MLP的新颖性。\n\n',
        tables['decision'].to_markdown(index=False) + '\n\n',
        quality.to_markdown(index=False) + '\n\n',
        '![独立比较](data/context_reference_confirm_20260912/quality_comparison.png)\n\n',
        '所有臂每图128次full、0次额外prefix，Context读出使用旧3000步最终EMA。'
        '原IG、ADG、Strong不安装新增捕获hook，Context才安装。质量采样秒数含采样和解码，'
        '不含加载、预检、CPU特征提取或审计。\n\n',
        timing.to_markdown(index=False) + '\n\n',
        '上表是上一阶段对相同EMA及运行时所做的同卡3次计时，清楚标记复用来源；并非本次重新测量。'
        '相同主干调用数不等于相同墙钟预算，若继续扩大验证，须把原IG增加相近计算的对照纳入。\n\n',
        '## 解释与不能得出的结论\n\n',
        'Local与Context原本用相同读出结构和训练批次，差别是是否读取跨patch计算后的表示。'
        'Local失败说明这一固定信息限制没有提供可靠的改进方向。Context与原IG又同时改变了读出结构和训练过程，'
        '即使数值下降，也不能唯一归因于“更准确的弱模型”“更匹配的共同误差”或某个频率机制。\n\n',
        '加性预测误差的分析仍给出约束：强弱共同比例的主要偏差可能被对比抵消，参考特有预测误差则会随引导幅度放大。'
        '本轮没有独立识别这些误差项，因此不以公式替代生成证据，也不把验证MSE下降当成方法成功。\n\n',
        '[ADG](https://arxiv.org/html/2506.11039v1)是已有角度引导方法，本轮直接继承仓库函数并检查入口一致性。'
        '[SSG](https://arxiv.org/html/2607.29122v1)已研究冻结主干上的中间adapter，故此类结构本身不是新贡献；'
        '本轮没有采用其合成数据监督。当前仍未完成具有实质新意、解释性理论和可靠同预算收益的长期目标。\n\n',
        '![固定前四张](data/context_reference_confirm_20260912/first4.png)\n\n',
        '[冻结协议](CONTEXT_REFERENCE_CONFIRM_PROTOCOL_20260912_ZH.md) · '
        '[原Local筛选](IG_INPUT_LOCAL_RESULTS_20260912_ZH.md) · '
        '[源数据工作簿](data/context_reference_confirm_20260912/source_data.xlsx) · '
        '[核验记录](data/context_reference_confirm_20260912/verification.json)。\n']
    REPORT.write_text(''.join(text))
    labels = np.load(m.ROOT / 'sit_small' / m.STAGE / 'inputs/labels.npy')
    assert np.array_equal(np.bincount(labels, minlength=100), np.full(100, 10))
    wb = openpyxl.load_workbook(OUT / 'source_data.xlsx', read_only=True)
    assert wb['quality'].max_row == 7
    wb.close()
    c.atomic(OUT / 'verification.json', dict(passed=True, arms=6, images=6000, audits=audits,
        max_fid_recalculation_error=max(a['absolute_error'] for a in audits),
        independent_feature_extraction=False, ten_images_per_class_verified=True,
        passes_1k_resource_gate=passed, visual_inspection_pending=True, goal_complete=False))
    c.atomic(OUT / 'artifact_manifest.json', dict(
        files={str(p.relative_to(c.WORK)): c.sha(p) for p in OUT.iterdir() if p.name != 'artifact_manifest.json'},
        report_sha256=c.sha(REPORT), generator_sha256=c.sha(Path(__file__))))
    print('Finalized', REPORT, 'passes', passed, flush=True)


if __name__ == '__main__':
    finalize()
