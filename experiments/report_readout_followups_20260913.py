"""Audit and publish the bounded IG matched control and CFG null-readout tests."""
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
import torch

from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import audit, local_head as local, mixture


CASES = {
    'matched': ('ig_readout_matched_control_20260913', 'IG_READOUT_MATCHED_CONTROL_RESULTS_20260913_ZH.md'),
    'cfg': ('cfg_null_readout_20260913', 'CFG_NULL_READOUT_RESULTS_20260913_ZH.md'),
}


def verify_request(path):
    request = local.verify_request(path)
    request = c.read(path)
    checks = {}
    for group in ('sources', 'assets', 'data', 'inputs', 'heads', 'comparisons'):
        for source, digest in request.get(group, {}).items():
            assert c.sha(source) == digest, source
            checks[source] = digest
    return dict(path=str(path), sha256=c.sha(path), files=checks)


def table_sources(paths):
    return pd.DataFrame([dict(path=str(p), sha256=c.sha(p)) for p in sorted(set(paths))])


def figures(out, quality, roots, title, native, candidate):
    delta = quality.fid - native
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    ax.barh(range(len(quality)), delta,
            color=['#176b8d' if arm == candidate else '#999999' for arm in quality.arm])
    ax.set_yticks(range(len(quality)), quality.arm)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=.8)
    ax.set_xlabel('FID difference from original baseline; lower is better')
    ax.set_title(title)
    lo, hi = min(float(delta.min()), 0.) - 1.5, max(float(delta.max()), 0.) + 4
    ax.set_xlim(lo, hi)
    for i, row in enumerate(quality.itertuples()):
        ax.text(hi - .03 * (hi - lo), i, f'FID {row.fid:.3f}', va='center', ha='right')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out / ('quality_comparison.' + ext), dpi=180)
    plt.close(fig)
    canvas = Image.new('RGB', (185 + 4 * 192, 45 + 210 * len(quality)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    draw.text((10, 10), 'SiT-S/2 ImageNet-100: first four in order; no selection', font=font, fill='black')
    for i, row in enumerate(quality.itertuples()):
        with np.load(roots[row.arm] / 'samples.npz') as data:
            pixels = data['arr_0'][:4]
        draw.text((10, 70 + i * 210), row.arm, font=font, fill='black')
        draw.text((10, 95 + i * 210), f'FID {row.fid:.2f}', font=font, fill='black')
        for j, pixel in enumerate(pixels):
            canvas.paste(Image.fromarray(pixel).resize((192, 192)), (185 + 192 * j, 45 + 210 * i))
    canvas.save(out / 'first4.png')


def build(case):
    name, filename = CASES[case]
    root = c.EXPS / name
    out = c.WORK / 'docs/data' / name
    out.mkdir(parents=True, exist_ok=True)
    report = c.WORK / 'docs' / filename
    assert c.read(root / 'status.json')['phase'] == 'complete'
    training = c.read(root / 'sit_small/training/summary.json')
    benchmark = c.read(root / 'sit_small/inference_benchmark.json')
    assert training['complete'] and benchmark['complete']
    requests = [verify_request(root / 'sit_small/training/request.json')]
    source_files = [root / 'status.json', root / 'sit_small/training/summary.json',
                    root / 'sit_small/inference_benchmark.json']
    extra_tables = {}
    if case == 'matched':
        from experiments.ig_readout_matched_control_20260913 import core as m
        m.prepare_sampling()
        arms = ('raw', 'native_fresh', 'native_base', 'adg')
        candidate = 'raw'
        roots = {a: (root / 'sit_small' / m.STAGE / a if a == m.ARM else m.ORIGINAL / a) for a in arms}
        new_arms = [m.ARM]
        reused = [a for a in arms if a != m.ARM]
        input_checks = []
        for p in (root / 'sit_small' / m.STAGE / 'inputs').glob('*.npy'):
            other = m.ORIGINAL / 'inputs' / p.name
            assert c.sha(p) == c.sha(other)
            input_checks.append(dict(file=p.name, identical_sha256=c.sha(p)))
        checks = c.read(root / 'sit_small' / m.STAGE / 'checks.json')
        assert checks['passed'] and checks['direct_weak_readout_exact']
        requests.append(verify_request(m.ORIGINAL / 'request.json'))
        extra_tables['shared_inputs'] = pd.DataFrame(input_checks)
        direct_path = c.EXPS / 'context_reference_5k_20260912/sit_small/direct_readout_verification.json'
        direct = c.read(direct_path)
        assert direct['complete'] and direct['equal_latents_and_pixels']
        source_files.append(direct_path)
        extra_tables['retained_candidate_timing'] = pd.DataFrame([
            dict(arm=a, median_seconds=v, batch=direct['batch'], repeats=direct['repeats'])
            for a, v in direct['medians'].items()])
        title = 'IG原结构读出的等训练补充控制'
        plot_title = 'IG matched training control: paired 1K, retrospective'
        timing_native = 'native_original'
    else:
        from experiments.cfg_null_readout_20260913 import core as m
        m.prepare()
        arms, candidate = m.ARMS, 'mlp'
        roots = {a: root / 'sit_small' / m.STAGE / a for a in arms}
        new_arms, reused = list(arms), []
        checks = {str(i): c.read(root / 'sit_small' / m.STAGE / f'checks{i}.json') for i in range(2)}
        assert all(row['passed'] and row['zero_guidance_exact'] and row['conditional_outputs_exact']
                   and row['apg_full_trajectory_exact'] for row in checks.values())
        title = '纯CFG无条件读出的固定生成结果'
        plot_title = 'CFG null readout: fixed 400-image screen'
        timing_native = 'native_base'
    request_path = root / 'sit_small' / m.STAGE / 'request.json'
    requests.append(verify_request(request_path))
    source_files += [m.PROTOCOL, request_path, root / 'sit_small/training/request.json']
    rows = {}
    audits = []
    torch.set_num_threads(4)
    for arm in arms:
        arm_root = roots[arm]
        summary = c.read(arm_root / 'summary.json')
        metric = c.read(arm_root / 'metrics.json')
        assert summary['complete'] and summary['primary_samples'] == m.N
        assert c.sha(arm_root / 'samples.npz') == summary['samples_sha256'] == metric['samples_sha256']
        row = {k: summary[k] for k in ('primary_samples', 'generated_paths',
                    'full_calls_per_output', 'prefix_calls_at_inference', 'seconds')}
        row.update(arm=arm, fid=metric['fid'], inception_score=metric['inception_score'],
                   newly_generated=arm in new_arms, stored_at=str(arm_root))
        rows[arm] = row
        mixture.ROOT = arm_root.parents[2]
        result = audit.arm('sit_small', arm_root.parent.name, arm)
        assert c.sha(arm_root / 'activations.npz') == result['features_sha256']
        assert c.sha(audit.REFS['sit_small']) == result['reference_sha256']
        audits.append(dict(reused_previous_audit=arm in reused, **result))
        source_files += [arm_root / f for f in ('summary.json', 'metrics.json', 'audit.json')]
    native_fid = rows['native_base']['fid']
    quality = pd.DataFrame([dict(**rows[a], delta_from_original=rows[a]['fid'] - native_fid) for a in arms])
    timing = pd.DataFrame([dict(arm=a, median_seconds=v,
        min_seconds=min(benchmark['seconds'][a]), max_seconds=max(benchmark['seconds'][a]),
        relative_to_original=v / benchmark['medians'][timing_native] - 1,
        batch=benchmark['batch'], repeats=benchmark['repeats'])
        for a, v in benchmark['medians'].items()])
    timing_repeats = pd.DataFrame([dict(arm=a, repeat=i + 1, seconds=v)
        for a, values in benchmark['seconds'].items() for i, v in enumerate(values)])
    training_table = pd.DataFrame([dict(arm=a, validation_mse=v,
        parameters=training['parameters'][a], steps=training['steps'],
        shared_loop_seconds=training['shared_training_seconds'])
        for a, v in training['validation_mse'].items()])
    if case == 'matched':
        raw, fresh = rows['raw'], rows['native_fresh']
        passes = raw['fid'] <= fresh['fid'] - 1 and raw['inception_score'] >= .9 * fresh['inception_score']
        training_sufficient = fresh['fid'] <= raw['fid'] + .5
        assert training['raw_replay_exact'] and training['raw_replay_max_difference'] == 0
        decision = dict(passes_readout_bundle_gate=passes, training_alone_sufficient_by_fixed_gate=training_sufficient,
            raw_minus_fresh=raw['fid'] - fresh['fid'], fresh_minus_old=fresh['fid'] - native_fid,
            retrospective_control=True, independent_confirmation=False, new_images=1000, reused_images=3000,
            unique_nonlinearity_explanation=False, mixture_model_identified=False, goal_complete=False)
        intro = ('相同3000步训练后，原结构新头FID为65.7192，保留MLP为63.7752；'
            'MLP仍低1.9440并满足IS门槛。因此训练配置变化没有解释掉全部差异，'
            '整套读出结构与输入处理应继续保留为研究因素。原结构重训也比旧头67.0089改善1.2896，'
            '不能把原来的全部收益归于结构，更不能将这些FID差值作可加的因果贡献分配。')
        design = ('这是看到先前生成结果后的补充控制。只新生成原结构新头1000图，复用已保存Raw、'
            '原IG及ADG各1000图；种子2026121341，第一噪声、备用第二噪声及标签文件逐个SHA一致。'
            '第二噪声不参与生成，每类10图、每图一条路径。它不是新独立确认，也不是4000张新图。'
            '原结构保留AdaLN+Linear和原条件输入，从官方零初始化开始；MLP同批次重放最终EMA逐tensor等于保留权重。'
            '强模型、depth4、真实数据、clean/噪声/时间、batch32、优化器、EMA与IG设置均固定。'
            '两头的输入标准化、参数化和位置输入仍不同，不能唯一归因于MLP非线性。')
        cost = ('所有质量比较均为64步Heun、128次full、0额外prefix。原结构新旧头均301,840参数，'
            '下表只比较本轮同卡的新旧原结构完整采样与解码。保留MLP为304,528参数，'
            '直接替换原弱头后净增2,688；另一组已核验同卡计时相对原IG增加0.29%，'
            '24条完整轨迹与原捕获实现像素及latent逐项一致。两批计时独立，不跨批相除。'
            '[可复用入口与成本记录](CONTEXT_REFERENCE_DIRECT_READOUT_20260913_ZH.md)保留全部证据。')
        limitation = ('本检验通过的是固定效应门槛，没有估计多训练种子置信区间。3000步是固定比较预算，'
            '未证明两个优化问题都已收敛，未区分结构容量、优化速度、位置输入及输入处理。'
            '验证MSE只记录，不等同采样质量。它也没有识别共同污染系数或加性误差场。'
            '[此前独立5K](CONTEXT_REFERENCE_5K_RESULTS_20260912_ZH.md)仍支持具体SiT读出收益，'
            '[归一化检验](IG_READOUT_NORMALIZATION_RESULTS_20260912_ZH.md)没有支持统计丢失解释，'
            'RAEv2的原400图也未见相同收益。[SSG](https://arxiv.org/html/2607.29122v1)'
            '已有冻结中间adapter的先例，不能把小MLP本身包装成核心新意。')
        literature = [dict(title='SSG', location='https://arxiv.org/html/2607.29122v1',
                          role='Prior intermediate adapter; not an identification of our mechanism')]
    else:
        best_control = min((rows[a] for a in arms if a != candidate), key=lambda x: x['fid'])
        margin = rows[candidate]['fid'] - best_control['fid']
        passes = margin <= -2 and rows[candidate]['inception_score'] >= .9 * rows['native_base']['inception_score']
        decision = dict(passes_frozen_screen_gate=passes, mlp_minus_native=rows[candidate]['fid'] - native_fid,
            best_control=best_control['arm'], mlp_minus_best_control=margin,
            advance_to_1k=passes, new_images=2800, reused_images=0,
            stop_this_construction=not passes, universal_null_readout_claim=False, goal_complete=False)
        intro = ('三个重新训练的无条件读出都没有优于原CFG。本轮MLP FID为77.9967，'
            '原CFG为77.4335，既有APG适配为76.6516。MLP比原CFG高0.5632，'
            '比最佳控制高1.3452，未达到比全部六个控制至少低2的预设门槛。'
            '停止这项构造，不进入1K，也不追加头宽、深度、学习率、训练时长或guidance强度搜索。')
        design = ('七臂各400张新图，共2800张，种子2026121361、每类4图、同噪声和标签。'
            '三种null头只读完整12层null特征，用相同真实数据FM训练3000步；'
            '分别为MLP从零输出初始化、原AdaLN从官方零初始化、原AdaLN从预训练末层续训。'
            '条件分支及全部主干冻结，训练不使用strong预测或生成数据。'
            '所有新头和原CFG额外系数1.25，半强度为0.625；沿用left_time<0.75窗口。'
            '第二噪声文件不使用，每图只有一条路径。')
        cost = ('全部七臂为64步Heun、224次full、0额外prefix。CFG仍需原条件末层，'
            '因此MLP实际额外保存310,688参数，两个AdaLN控制各308,000；不能只报新旧头之差2,688。'
            '同卡batch8、轮换预热后各三次完整采样及解码，中位数MLP相对原CFG增加约0.69%。'
            '但第三轮多臂出现约2秒长尾，明显高于约1.25秒常态；保留全部逐次值，'
            '不据这组三次计时给出精确开销结论。失败构造不再追加速度调优。')
        limitation = ('400图是固定小规模筛选，不能凭小FID差宣称总体显著变差；'
            '确定的是没有达到预先要求的实际改善。这个失败不能证明所有CFG无条件分支都不值得改进，'
            '也不能把SiT浅层IG读出的收益推广到完整主干后的null读出。'
            '本轮APG为仓库既有cfg_apg_07适配：强度2、velocity差的负动量-0.5、'
            '截断与clean方向投影；不冒充论文clean历史的逐项等价实现。'
            '新代码与既有实现的完整轨迹逐项相同，Heun两stage读取同一旧历史并提交第一stage历史。'
            '条件输出不变、null直接读出及零guidance退回原模型检查均通过。')
        literature = [
            dict(title='Unconditional Priors Matter', location='https://arxiv.org/html/2503.20240v2',
                 role='Motivation in fine-tuned models; second-backbone replacement differs from this experiment'),
            dict(title='APG', location='https://arxiv.org/html/2410.02416v2',
                 role='Existing guidance; our control retains the repository velocity-history adaptation'),
            dict(title='ADG', location='https://arxiv.org/html/2506.11039v1', role='Existing angular guidance control')]
        limitation += ('[Unconditional Priors Matter](https://arxiv.org/html/2503.20240v2)'
            '主要在微调模型中用另一模型改善无条件先验，它不能证明这里的末层读出是瓶颈。'
            '[APG](https://arxiv.org/html/2410.02416v2)与[ADG](https://arxiv.org/html/2506.11039v1)'
            '提供既有比较方法，本轮没有新增可用的纯CFG方法。')
        source_files.append(c.WORK / 'readings/cfg_null_readout_20260913/source_manifest.json')
    c.atomic(out / 'decision.json', decision)
    c.atomic(root / 'decision.json', decision)
    tables = dict(quality=quality, decision=pd.DataFrame([decision]), timing=timing,
                  timing_repeats=timing_repeats, training=training_table,
                  training_history=pd.DataFrame(training['history']), **extra_tables)
    tables['Sources'] = pd.DataFrame([
        dict(title='Frozen protocol', location=str(m.PROTOCOL), role='Arms, data, budget and stopping rule'),
        dict(title='Raw outputs', location=str(root), role='Saved images, requests and original evaluation'),
        dict(title='Additive-error hypothesis', location=str(c.WORK / 'docs/AG_IG_STRUCTURED_MISMATCH_20260912_ZH.md'),
             role='Retained theoretical structure; not identified by these experiments'), *literature])
    tables['source_files'] = table_sources(source_files)
    for table_name, table in tables.items():
        table.to_csv(out / (table_name + '.csv'), index=False)
    with pd.ExcelWriter(out / 'source_data.xlsx', engine='openpyxl') as writer:
        for table_name, table in tables.items():
            table.to_excel(writer, sheet_name=table_name, index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = min(70, max(16, max(len(str(x.value or '')) for x in col) + 2))
    figures(out, quality, roots, plot_title, native_fid, candidate)
    rel = 'data/' + name + '/'
    body = [f'# {title}\n\n', intro + '\n\n', design + '\n\n',
            quality[['arm', 'fid', 'inception_score', 'newly_generated', 'full_calls_per_output']].to_markdown(index=False) + '\n\n',
            f'![固定质量比较]({rel}quality_comparison.png)\n\n', cost + '\n\n',
            timing.to_markdown(index=False) + '\n\n',
            '共享训练循环时间不含加载与最终验证；验证MSE未用于挑checkpoint或调整方案。\n\n',
            training_table.to_markdown(index=False) + '\n\n', limitation + '\n\n',
            f'![固定前四张]({rel}first4.png)\n\n',
            '全部新生成样本的逐批图像、标签、噪声、请求及源码/权重SHA和调用数已核对，'
            'FID从同一Inception缓存特征用FP64复算；这不是独立特征提取器验证。'
            '复用对照沿用已核验FID，同时重新校验样本、指标、特征与参考统计SHA。\n\n',
            f'[冻结协议]({m.PROTOCOL.name}) · [源数据工作簿]({rel}source_data.xlsx) · '
            f'[判定]({rel}decision.json) · [核验记录]({rel}verification.json)。\n']
    report.write_text(''.join(body))
    labels = np.load(root / 'sit_small' / m.STAGE / 'inputs/labels.npy')
    np.testing.assert_array_equal(np.bincount(labels, minlength=100), np.full(100, m.N // 100))
    verification = dict(passed=True, new_arms=len(new_arms), new_images=len(new_arms) * m.N,
        reused_arms=reused, reused_images=len(reused) * m.N, all_labels_balanced=True, requests=requests,
        preflight=checks, audits=audits, max_fid_recalculation_error=max(r['absolute_error'] for r in audits),
        independent_feature_extraction=False, visual_inspection_pending=True, goal_complete=False)
    c.atomic(out / 'verification.json', verification)
    c.atomic(out / 'artifact_manifest.json', dict(report_sha256=c.sha(report), generator_sha256=c.sha(Path(__file__)),
        files={str(p.relative_to(c.WORK)): c.sha(p) for p in out.iterdir() if p.name != 'artifact_manifest.json'}))
    print('Built', case, decision, flush=True)


def review(case):
    name, filename = CASES[case]
    out = c.WORK / 'docs/data' / name
    report = c.WORK / 'docs' / filename
    manifest = c.read(out / 'artifact_manifest.json')
    assert c.sha(report) == manifest['report_sha256']
    assert c.sha(Path(__file__)) == manifest['generator_sha256']
    for filename, digest in manifest['files'].items():
        assert c.sha(c.WORK / filename) == digest, filename
    tables = []
    wb = openpyxl.load_workbook(out / 'source_data.xlsx', read_only=True, data_only=True)
    for sheet in wb.worksheets:
        with (out / (sheet.title + '.csv')).open() as stream:
            expected = list(csv.reader(stream))
        actual = list(sheet.values)
        assert len(actual) == len(expected), sheet.title
        for row, want in zip(actual, expected):
            assert len(row) == len(want)
            for x, y in zip(row, want):
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
    for link in re.findall(r'\]\(([^)]+)\)', report.read_text()):
        if link.startswith(('https://', 'http://', '#')):
            continue
        assert (report.parent / link).exists(), link
        links.append(link)
    v = c.read(out / 'verification.json')
    assert v['passed']
    v.update(visual_inspection_pending=False, workbook_tables_reconciled=tables, local_links_verified=links,
             inspected_figures={p.name: c.sha(p) for p in out.glob('*.png')},
             visual_notes='All listed PNGs displayed and manually checked; readable labels and fixed sample order.')
    c.atomic(out / 'verification.json', v)
    manifest['files'] = {str(p.relative_to(c.WORK)): c.sha(p) for p in out.iterdir() if p.name != 'artifact_manifest.json'}
    c.atomic(out / 'artifact_manifest.json', manifest)
    print('Finalized', case, tables, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--visuals-reviewed', action='store_true')
    args = parser.parse_args()
    if args.visuals_reviewed:
        review(args.case)
    else:
        build(args.case)
