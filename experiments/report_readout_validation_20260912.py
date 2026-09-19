"""Finalize paired generation, source workbooks, cost records and falsifiable decisions."""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import audit, mixture


def fid_covariance(features, mu, covariance):
    """FP64 symmetric covariance form; avoid a rank-deficient 5000-by-5000 Gram matrix."""
    x = np.asarray(features, dtype=np.float64)
    center = x.mean(0)
    x = x - center
    cov = x.T @ x / (len(x) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh((cov + cov.T) * .5)
    assert eigenvalues.min() > -1e-6
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    middle = root @ np.asarray(covariance, dtype=np.float64) @ root
    values = np.linalg.eigvalsh((middle + middle.T) * .5)
    assert values.min() > -1e-6
    return float(np.square(center - mu).sum() + np.trace(cov) + np.trace(covariance)
                 - 2 * np.sqrt(np.maximum(values, 0)).sum())


def wait_complete(m):
    while True:
        state = c.read(m.ROOT / 'status.json')
        if state['phase'] == 'complete':
            return
        if state['phase'] == 'failed' or not Path('/proc', str(state['pid'])).exists():
            raise RuntimeError('Controller ended before completion')
        time.sleep(5)


def finalize(case):
    if case == '5k':
        from experiments.context_reference_5k_20260912 import core as m
        report = c.WORK / 'docs/CONTEXT_REFERENCE_5K_RESULTS_20260912_ZH.md'
        title = '上下文读出的独立5K与成本验证'
        candidate = 'context_base'
    else:
        from experiments.ig_readout_normalization_20260912 import core as m
        from experiments.ig_readout_normalization_20260912 import train as tr
        report = c.WORK / 'docs/IG_READOUT_NORMALIZATION_RESULTS_20260912_ZH.md'
        title = '归一化统计解释的真实生成检验'
        candidate = 'raw'
    out = c.WORK / 'docs/data' / m.ROOT.name
    out.mkdir(parents=True, exist_ok=True)
    wait_complete(m)
    m.configure()
    m.verify('sit_small')
    results = c.read(m.ROOT / 'sit_small' / m.STAGE / 'results.json')
    assert len(results) == len(m.ARMS)
    by = {r['arm']: r for r in results}
    mixture.ROOT = m.ROOT
    torch.set_num_threads(4)
    if case == '5k':
        audit.fid_from_features = fid_covariance
    audits = [audit.arm('sit_small', m.STAGE, arm) for arm in m.ARMS]
    quality = pd.DataFrame([{k: by[arm][k] for k in ('arm', 'fid', 'inception_score', 'seconds',
        'primary_samples', 'generated_paths', 'full_calls_per_output', 'prefix_calls_at_inference')} for arm in m.ARMS])
    benchmark = c.read(m.ROOT / 'sit_small/inference_benchmark.json')
    assert benchmark['complete']
    timing = pd.DataFrame([dict(arm=arm, median_seconds=seconds, batch=benchmark['batch'],
        repeats=benchmark['repeats'], seconds_relative_to_native=seconds / benchmark['medians']['native_base'] - 1)
        for arm, seconds in benchmark['medians'].items()])
    supplemental = {}
    if case == '5k':
        best = min((by[a] for a in m.ARMS if a != candidate), key=lambda r: r['fid'])
        passed = by[candidate]['fid'] <= best['fid'] - 1 and by[candidate]['inception_score'] >= .9 * by['native_base']['inception_score']
        decision = dict(candidate_fid=by[candidate]['fid'], best_control=best['arm'], best_control_fid=best['fid'],
            delta=by[candidate]['fid'] - best['fid'], native_delta=by[candidate]['fid'] - by['native_base']['fid'],
            adg_delta=by[candidate]['fid'] - by['adg']['fid'], native66_delta=by[candidate]['fid'] - by['native_66']['fid'],
            passes_frozen_5k_gate=passed, wall_time_control_covers_capture=benchmark['native66_relative_to_context'] >= 0,
            statistical_significance_established=False, cross_model_success=False, novelty_established=False, goal_complete=False)
        conclusion = ('这次具体上下文读出通过独立5K的固定效应门槛，包括更高主干预算的原IG对照。'
                      if passed else '这次具体上下文读出未通过独立5K的固定效应门槛，不追加参数搜索。')
        explanation = ('候选最初是Local试验中的Context对照，在400图中被选中，随后在独立1K通过门槛，'
            '再在这组全新5000噪声上确认。训练、checkpoint、引导强度和时间窗口均未改变。'
            '原IG与ADG是同128次full对照；原IG66步是132次full的成本对照，分段窗口的离散落点也随网格变化。'
            '四组都只用一条生成路径，每类50图。质量采样秒数来自并行GPU工作者，不能用来判定同卡速度。')
        limitation = ('通过该门槛不等于统计显著性检验，也没有证明混合模型或加性误差机制。'
            'RAEv2的原400图试验没有同样收益，因此不能声称跨模型成立。冻结中间读出已有先例，MLP结构本身不是新贡献。')
        old = c.read(c.EXPS / 'context_reference_confirm_20260912/sit_small/confirm_1000/results.json')
        supplemental['selection_1k'] = pd.DataFrame([{k: r[k] for k in ('arm', 'fid', 'inception_score')} for r in old])
        direct_path = m.ROOT / 'sit_small/direct_readout_verification.json'
        if direct_path.exists():
            direct = c.read(direct_path)
            assert direct['complete'] and direct['equal_latents_and_pixels']
            supplemental['direct_timing'] = pd.DataFrame([dict(arm=arm, median_seconds=value,
                batch=direct['batch'], repeats=direct['repeats']) for arm, value in direct['medians'].items()])
            supplemental['direct_equivalence'] = pd.DataFrame(direct['checks'])
            direct_text = (f"额外完成了不改变采样公式的实现精简：在原共享前向中直接替换弱读出，"
                f"不再同时计算原弱头。固定24条完整轨迹的latent及解码像素逐项一致。"
                f"原弱头{direct['parameters']['native']:,}参数，新头{direct['parameters']['context']:,}参数，"
                f"替换后净增加{direct['net_added_parameters']:,}。该实现同卡中位耗时相对原IG"
                f"{direct['context_direct_relative_to_native']:+.2%}。这组计时单独报告，冻结5K采样代码未被修改。")
        else:
            direct_text = '直接替换弱头的等价实现检查尚未完成，本报告暂不使用其成本数据。'
    else:
        training = c.read(tr.TRAIN / 'summary.json')
        raw, ln, stats = (by[a] for a in tr.MODES)
        raw_effect = raw['fid'] <= min(by[a]['fid'] for a in ('native_base', 'adg')) - 1
        removed_hurts = ln['fid'] >= raw['fid'] + 1
        restored_recovers = stats['fid'] <= ln['fid'] - 1 and stats['fid'] <= raw['fid'] + .5
        is_ok = stats['inception_score'] >= .9 * raw['inception_score']
        passed = raw_effect and removed_hurts and restored_recovers and is_ok
        decision = dict(raw_fid=raw['fid'], ln_fid=ln['fid'], ln_stats_fid=stats['fid'],
            ln_minus_raw=ln['fid'] - raw['fid'], stats_minus_ln=stats['fid'] - ln['fid'],
            stats_minus_raw=stats['fid'] - raw['fid'], raw_beats_controls_by_one=raw_effect,
            removing_statistics_hurts_by_one=removed_hurts, restoring_statistics_recovers=restored_recovers,
            passes_fixed_mechanism_prediction=passed, single_training_seed=True,
            raw_replay_exact=training['raw_replay_exact'], causal_identification_complete=False, goal_complete=False)
        conclusion = ('固定生成对照支持归一化统计对这次读出差异有贡献，但不建立唯一因果解释。'
                      if passed else '固定生成对照没有支持“归一化统计解释当前收益”这一完整预测。')
        explanation = ('三个参考共享相同depth4特征、训练数据、噪声、时间、MLP宽度、训练3000步和最终EMA。'
            'Raw保留原特征，LN先做token内LayerNorm，LN+stats另输入均值与带epsilon的尺度。'
            '后者只增加768个权重，没有引导系数、局部门控或额外前缀。'
            'Raw最终EMA与旧Context逐tensor比较的结果也保留。每臂1000个新噪声、每类10图，'
            '同128次full，与原IG和ADG一起评估。')
        limitation = ('这是看到SiT/RAEv2差别后提出的解释候选。epsilon非零时不能宣称LayerNorm严格消除全部尺度信息；'
            '有限容量、优化和特征流形可能影响结果。单训练种子和单1K不提供普遍因果结论，'
            '训练MSE不作为方法成功或采样质量的替代。归一化统计的移除与恢复已有RevIN先例。')
        supplemental['training'] = pd.DataFrame([dict(mode=mode, validation_mse=training['validation_mse'][mode],
            parameters=training['parameters'][mode], steps=training['steps'], shared_loop_seconds=training['training_seconds'],
            raw_replay_max_difference=training['raw_replay_max_difference']) for mode in tr.MODES])
        supplemental['training_history'] = pd.DataFrame(training['history'])
        direct_text = ''
    tables = dict(quality=quality, decision=pd.DataFrame([decision]), timing=timing, **supplemental)
    for name, table in tables.items():
        table.to_csv(out / (name + '.csv'), index=False)
    c.atomic(out / 'decision.json', decision)
    sources = pd.DataFrame([
        dict(title='Frozen protocol', location=str(m.PROTOCOL), role='Predefined sample bank, arms, predictions and stopping rules'),
        dict(title='Raw outputs', location=str(m.ROOT), role='Images, labels, hashes, counts and original evaluator features'),
        dict(title='Structured mismatch and additive errors', location=str(c.WORK / 'docs/AG_IG_STRUCTURED_MISMATCH_20260912_ZH.md'), role='Hypothesis retained; not identified by this experiment'),
        dict(title='ADG', location='https://arxiv.org/html/2506.11039v1', role='Existing comparison algorithm'),
        dict(title='SSG', location='https://arxiv.org/html/2607.29122v1', role='Prior frozen intermediate adapters; no synthetic training used here'),
        dict(title='RevIN official implementation', location='https://github.com/ts-kim/RevIN', role='Prior removal and restoration of instance statistics')])
    with pd.ExcelWriter(out / 'source_data.xlsx', engine='openpyxl') as writer:
        for name, table in dict(tables, Sources=sources).items():
            table.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = min(65, max(16, max(len(str(v.value or '')) for v in col) + 2))
    figure, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    delta = quality.fid - by['native_base']['fid']
    ax.barh(range(len(quality)), delta, color=['#176b8d' if a == candidate else '#999999' for a in quality.arm])
    ax.set_yticks(range(len(quality)), quality.arm)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=.8)
    ax.set_xlabel('FID difference from native IG; lower is better')
    ax.set_title('SiT-S/2: independent 5K confirmation' if case == '5k' else 'SiT-S/2: fixed normalization test, independent 1K')
    lo, hi = float(delta.min()) - 2, float(delta.max()) + 5
    ax.set_xlim(lo, hi)
    for i, row in enumerate(quality.itertuples()):
        ax.text(hi - .02 * (hi - lo), i, f'FID {row.fid:.3f}', va='center', ha='right')
    for ext in ('png', 'pdf', 'svg'):
        figure.savefig(out / ('quality_comparison.' + ext), dpi=180)
    plt.close(figure)
    canvas = Image.new('RGB', (185 + 4 * 192, 45 + 210 * len(m.ARMS)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    draw.text((10, 10), f'SiT-S/2: {m.N} per arm; first four in order; no selection', font=font, fill='black')
    for i, arm in enumerate(m.ARMS):
        with np.load(m.ROOT / 'sit_small' / m.STAGE / arm / 'samples.npz') as data:
            images = data['arr_0'][:4]
        draw.text((10, 70 + i * 210), arm, font=font, fill='black')
        draw.text((10, 95 + i * 210), f"FID {by[arm]['fid']:.2f}", font=font, fill='black')
        for j, pixels in enumerate(images):
            canvas.paste(Image.fromarray(pixels).resize((192, 192)), (185 + 192 * j, 45 + 210 * i))
    canvas.save(out / 'first4.png')
    relative = 'data/' + out.name + '/'
    pieces = [f'# {title}\n\n', conclusion + '\n\n', explanation + '\n\n',
        quality[['arm', 'fid', 'inception_score', 'primary_samples', 'full_calls_per_output']].to_markdown(index=False) + '\n\n',
        f'![比较]({relative}quality_comparison.png)\n\n', '固定判定：\n\n']
    pieces.extend(f'- {key}: {value}\n' for key, value in decision.items())
    pieces += ['\n同卡轮换计时包含完整采样及解码，加载与审计另计。\n\n', timing.to_markdown(index=False) + '\n\n']
    if direct_text:
        pieces += [direct_text + '\n\n']
    if 'direct_timing' in supplemental:
        pieces += [supplemental['direct_timing'].to_markdown(index=False) + '\n\n']
    if 'training' in supplemental:
        pieces += [supplemental['training'].to_markdown(index=False) + '\n\n']
    pieces += [limitation + '\n\n',
        '[ADG](https://arxiv.org/html/2506.11039v1)是既有基线；[SSG](https://arxiv.org/html/2607.29122v1)'
        '已研究冻结中间adapter；[RevIN官方代码](https://github.com/ts-kim/RevIN)提供恢复实例统计的相关先例。'
        '这些先例用于限定贡献边界，不代表它们证明本实验的解释。\n\n',
        f'![固定前四张]({relative}first4.png)\n\n',
        '审计核对全部标签、噪声和源码/权重hash、单条生成路径及调用数，并从同一缓存特征FP64复算FID。'
        '它不构成独立特征提取器验证。\n\n',
        f'[冻结协议]({m.PROTOCOL.name}) · [源数据工作簿]({relative}source_data.xlsx) · '
        f'[核验记录]({relative}verification.json)。\n']
    report.write_text(''.join(pieces))
    labels = np.load(m.ROOT / 'sit_small' / m.STAGE / 'inputs/labels.npy')
    assert np.array_equal(np.bincount(labels, minlength=100), np.full(100, m.N // 100))
    c.atomic(out / 'verification.json', dict(passed=True, arms=len(m.ARMS), images=len(m.ARMS) * m.N,
        all_labels_balanced=True, audits=audits, independent_feature_extraction=False,
        fid_recalculation_method='symmetric_fp64_covariance' if case == '5k' else 'fp64_gram',
        max_fid_recalculation_error=max(a['absolute_error'] for a in audits), visual_inspection_pending=True, goal_complete=False))
    c.atomic(out / 'artifact_manifest.json', dict(files={str(p.relative_to(c.WORK)): c.sha(p)
        for p in out.iterdir() if p.name != 'artifact_manifest.json'}, report_sha256=c.sha(report), generator_sha256=c.sha(Path(__file__))))
    print('Report complete', report, decision, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('5k', 'normalization'), required=True)
    args = parser.parse_args()
    finalize(args.case)
