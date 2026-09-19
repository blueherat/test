"""Report the completed input-local test, preserving all stronger controls."""
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
from experiments.guidance_distribution_20260912 import audit, mixture, local_head as local
from . import core as m

OUT = c.WORK / 'docs/data/input_local_completion_20260912'
REPORT = c.WORK / 'docs/IG_INPUT_LOCAL_RESULTS_20260912_ZH.md'
NAMES = {'sit_small': 'SiT-S/2', 'raev2': 'RAEv2'}


def finalize():
    m.configure()
    mixture.ROOT = m.ROOT
    torch.set_num_threads(4)
    while c.read(m.ROOT / 'status.json')['phase'] != 'complete':
        state = c.read(m.ROOT / 'status.json')
        if not Path('/proc', str(state['pid'])).exists():
            raise RuntimeError('Controller exited before all arms completed')
        time.sleep(5)
    for model in c.MODELS:
        while not (m.ROOT / model / 'inference_benchmark.json').exists():
            time.sleep(5)
    OUT.mkdir(parents=True, exist_ok=True)
    quality, decisions, training, timing, audits = [], [], [], [], []
    for model in c.MODELS:
        m.verify(model)
        assert not (local.ROOT / model / local.STAGE).exists()
        results = c.read(m.ROOT / model / m.STAGE / 'results.json')
        assert len(results) == len(m.ARMS)
        by = {r['arm']: r for r in results}
        for r in results:
            quality.append({k: r[k] for k in ('model', 'arm', 'fid', 'inception_score', 'seconds', 'primary_samples',
                                             'generated_paths', 'full_calls_per_output', 'prefix_calls_at_inference')})
            audits.append(audit.arm(model, m.STAGE, r['arm']))
        controls = [r for r in results if not r['arm'].startswith('local')]
        best = min(controls, key=lambda r: r['fid'])
        for arm in ('local_base', 'local_half'):
            candidate = by[arm]
            decisions.append(dict(model=model, candidate=arm, candidate_fid=candidate['fid'],
                best_control=best['arm'], best_control_fid=best['fid'], delta=candidate['fid'] - best['fid'],
                native_fid=by['native_base']['fid'], context_same_strength_fid=by[arm.replace('local', 'context')]['fid'],
                inception_ratio=candidate['inception_score'] / by['native_base']['inception_score'],
                passes=candidate['fid'] <= best['fid'] - 2 and candidate['inception_score'] >= .9 * by['native_base']['inception_score']))
        train = c.read(m.training_root(model) / 'summary.json')
        training.append(dict(model=model, steps=train['steps'], batch=train['batch'],
                             training_seconds=train['training_seconds'], parameters_per_head=train['parameters_per_head'],
                             local_validation_mse=train['validation_mse']['local'], context_validation_mse=train['validation_mse']['context']))
        bench = c.read(m.ROOT / model / 'inference_benchmark.json')
        for arm in ('local_base', 'context_base'):
            timing.append(dict(model=model, arm=arm, batch=bench['batch'], repeats=bench['repeats'],
                native_seconds=bench['medians']['native_base'], candidate_seconds=bench['medians'][arm],
                relative_change=bench['relative_changes'][arm], extra_parameters=bench['extra_parameters_per_candidate']))
    tables = {name: pd.DataFrame(rows) for name, rows in [('quality', quality), ('decisions', decisions), ('training', training), ('timing', timing)]}
    sources = pd.DataFrame([
        dict(title='Original frozen input-local protocol', location=str(local.PROTOCOL), role='Information limit, matched heads, 3000-step training and generation gate'),
        dict(title='Completion protocol', location=str(m.PROTOCOL), role='New output root, retained final EMA, unchanged primary method'),
        dict(title='Raw experiment assets', location=str(m.ROOT), role='16 arms, 6400 individual paths and hash records'),
        dict(title='Sliding Window Guidance', location='https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/paper.pdf', role='Prior spatially limited reference guidance; not a new broad principle'),
        dict(title='Synthetic Self-Guidance', location='https://arxiv.org/html/2607.29122v1', role='Related frozen-backbone intermediate heads; synthetic training excluded in this test')])
    for name, df in tables.items():
        df.to_csv(OUT / (name + '.csv'), index=False)
    with pd.ExcelWriter(OUT / 'source_data.xlsx', engine='openpyxl') as writer:
        for name, df in dict(tables, Sources=sources).items():
            df.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = min(65, max(16, max(len(str(v.value or '')) for v in col) + 2))
    q = tables['quality']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, model in zip(axes, c.MODELS):
        df = q[q.model == model].set_index('arm').loc[list(m.ARMS)]
        delta = df.fid - df.loc['native_base'].fid
        ax.barh(range(len(df)), delta, color=['#176b8d' if arm.startswith('local') else '#999999' for arm in df.index])
        ax.set_yticks(range(len(df)), df.index)
        ax.invert_yaxis()
        ax.axvline(0, color='black', lw=.8)
        ax.set_xlabel('FID difference from native IG; lower is better')
        ax.set_title(NAMES[model] + ' · 400 images')
        lo, hi = min(-3, float(delta.min()) - 4), max(8, float(delta.max()) + 16)
        ax.set_xlim(lo, hi)
        for i, r in enumerate(df.itertuples()):
            ax.text(hi - .02 * (hi - lo), i, f'FID {r.fid:.2f}', ha='right', va='center', fontsize=9)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / ('quality_comparison.' + ext), dpi=180)
    plt.close(fig)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    for model in c.MODELS:
        arms = ['native_base', 'context_base', 'local_base', 'local_half']
        canvas = Image.new('RGB', (185 + 192 * 4, 45 + 210 * len(arms)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), NAMES[model] + ': first four in order; no selection', font=font, fill='black')
        for i, arm in enumerate(arms):
            root = m.ROOT / model / m.STAGE / arm
            with np.load(root / 'samples.npz') as data:
                pixels = data['arr_0'][:4]
            draw.text((10, 70 + i * 210), arm, font=font, fill='black')
            draw.text((10, 93 + i * 210), f"FID {c.read(root / 'metrics.json')['fid']:.2f}", font=font, fill='black')
            for j, image in enumerate(pixels):
                canvas.paste(Image.fromarray(image).resize((192, 192)), (185 + j * 192, 45 + i * 210))
        canvas.save(OUT / (model + '_first4.png'))
    eligible = tables['decisions'][tables['decisions'].passes]
    conclusion = ('所有Local候选均未通过冻结筛选，当前输入局部构造结束；未为Local启动1K／5K、重新训练或超参数搜索。'
                  if len(eligible) == 0 else '部分Local候选通过400图筛选，必须进入全新1K并比较ADG／APG，尚不能称为可靠收益。')
    txt = [
        '# 输入局部内部参考的真实生成检验\n\n',
        conclusion + '\n\n',
        '本轮补完用户粘贴文本中的备用方案：两模型的Local／Context读出此前均已用真实数据训练3000步，但生成被暂缓。'
        '本轮直接使用保留的最终EMA，在新输出目录完成原先固定的16臂、每臂400张、共6400张单路径图像。'
        '这不是新提出的idea，也没有替换共同污染与加性残差的主要分布假设。纯CFG目标仍在继续，本轮没有新的CFG生成候选。\n\n',
        tables['decisions'].to_markdown(index=False) + '\n\n',
        'SiT的Context对照在400图上比原IG最好幅度低约2.97 FID，RAEv2的Context没有超过原IG。'
        'SiT这一意外信号已按[单独冻结的事后选择协议](CONTEXT_REFERENCE_CONFIRM_PROTOCOL_20260912_ZH.md)进入全新1K，'
        '加入原IG三档、ADG和Strong。它不改变本轮Local失败与无晋级的判断，也不能据此宣称原候选成功。\n\n',
        '![全部固定对照](data/input_local_completion_20260912/quality_comparison.png)\n\n',
        'Local只读取第一次attention前的空间token、原始时间／类别条件和固定位置编码。Context读取第4／8层后的token，'
        '使用完全相同的读出结构、初始参数、训练数据与批次。两者的差别是可用上下文，实际拟合难度仍不同。\n\n',
        '在理想总体MSE最优、表示能够保留对应输入信息的条件下，局部clean预测为 '
        '$E[X_j\\mid Z_j,t,c,j]$，与各patch条件边缘的乘积分布相容。强弱score之差因此可强调跨patch依赖。'
        '有限容量、有限训练头及冻结主干误差都不保证实现该密度关系。RAE的单token已是语义latent，不能把它称为原图局部感受野。'
        '该假说不意味着弱分布是强分布的高斯平滑，也不自动满足共同污染的混合关系。\n\n',
        '以质量对照检验信息限制：只有胜过Context、原IG不同幅度和Strong全部6个非Local对照至少2 FID、'
        '且IS保留原IG的90%，才允许进入新1K。Local仅优于Context不能作为成功。400图只是筛选，'
        'RAE仅覆盖400类，不能用其小幅FID变化支持完整ImageNet的统计结论。\n\n',
        '## 全部生成与成本\n\n', q.to_markdown(index=False) + '\n\n',
        '每图SiT为128次full、RAEv2为100次full，额外prefix为0。全部样本单独进入主评价，不做图像挑选或粒子重复。'
        '原生弱输出与Strong来自原共享前向；Local／Context额外读出只有被使用时才求值。\n\n',
        tables['timing'].to_markdown(index=False) + '\n\n',
        '计时为同卡、同批量，预热后各3次完整采样和解码。原IG不安装捕获hook，候选安装，故相对耗时包含这项实现开销。'
        '额外头参数、内存操作和延迟已单列，没有把相同主干NFE表述成完全零成本。\n\n',
        '## 保留的训练与实现检查\n\n', tables['training'].to_markdown(index=False) + '\n\n',
        '训练秒数是历史两头联合训练循环耗时，不是本轮新增训练，也不含模型加载和数据准备。'
        '训练请求的源文件、数据及checkpoint哈希已重新核验。模型参数仍冻结；hook不改变原Strong／Weak输出；'
        '只改目标patch外的输入时Local该patch预测逐位不变；共享捕获与直接embedding／prefix计算逐位一致；零引导恢复Strong。'
        '这些检查验证实现与成本，不证明生成机制或总体密度假说。\n\n',
        '## 已有工作与取舍\n\n',
        '[Sliding Window Guidance](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_26/paper.pdf)已有通过限制远程输入构造参考的路线。'
        '本轮使用共享embedding上的局部读出，不认领“局部参考”本身。'
        '[SSG预印本](https://arxiv.org/html/2607.29122v1)研究冻结像素模型、训练中间adapter并用其自身生成样本监督；'
        '这也排除了将冻结主干加小头当作普遍新贡献。本轮只用真实数据，未恢复已停止的合成数据／蒸馏路线。\n\n',
        '本轮结果与既有共同污染反演、反射交替分别记录，不用失败候选之间的局部胜出缩小长期成功要求。'
        '仍须取得有实质新意、解释性理论和可靠同预算收益的核心方法，长期目标未完成。\n\n',
        '![SiT固定前四个](data/input_local_completion_20260912/sit_small_first4.png)\n\n',
        '![RAEv2固定前四个](data/input_local_completion_20260912/raev2_first4.png)\n\n',
        '[原协议](IG_INPUT_LOCAL_PROTOCOL_20260912_ZH.md) · [补完协议](IG_INPUT_LOCAL_COMPLETION_20260912_ZH.md) · '
        '[源数据工作簿](data/input_local_completion_20260912/source_data.xlsx) · '
        '[核验记录](data/input_local_completion_20260912/verification.json) · '
        '[原文归档](../readings/input_local_completion_20260912/source_manifest.json)。\n']
    REPORT.write_text(''.join(txt))
    wb = openpyxl.load_workbook(OUT / 'source_data.xlsx', read_only=True)
    assert wb['quality'].max_row == 17
    wb.close()
    c.atomic(OUT / 'verification.json', dict(passed=True, arms=16, images=6400, additional_training_steps=0,
        audits=audits, max_fid_recalculation_error=max(r['absolute_error'] for r in audits),
        independent_feature_extraction=False, old_output_root_unchanged=True,
        eligible_for_1k=eligible[['model', 'candidate']].to_dict(orient='records'),
        visual_inspection_pending=True, goal_complete=False))
    c.atomic(OUT / 'artifact_manifest.json', dict(
        files={str(p.relative_to(c.WORK)): c.sha(p) for p in OUT.iterdir() if p.name != 'artifact_manifest.json'},
        report_sha256=c.sha(REPORT), generator_sha256=c.sha(Path(__file__))))
    print('Finalized', REPORT, 'eligible', eligible[['model', 'candidate']].to_dict(orient='records'), flush=True)


if __name__ == '__main__':
    finalize()
