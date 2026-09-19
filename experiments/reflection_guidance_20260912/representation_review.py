"""Audit existing-path render controls without changing candidate selection."""
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
from experiments.guidance_pasted_20260912.audit import REFS, fid_from_features
from experiments.guidance_distribution_20260912.audit import cached_features
from . import core as m, render_audit as render

OUT = c.WORK / 'docs/data/reflection_representation_20260912'
REPORT = c.WORK / 'docs/REFLECTION_REPRESENTATION_REVIEW_20260912_ZH.md'
NAMES = {'sit_small': 'SiT-S/2', 'raev2': 'RAEv2'}


def audit_render(model, track):
    root = m.ROOT / model / render.STAGE / (track + '_physical_flip')
    summary = c.read(root / 'summary.json')
    metrics = c.read(root / 'metrics.json')
    source = Path(summary['source_summary'])
    assert summary['complete'] and summary['diagnostic_only']
    assert summary['generated_paths'] == summary['incremental_full_calls'] == summary['incremental_prefix_calls'] == 0
    assert c.sha(source) == summary['source_summary_sha256']
    assert c.read(source)['records'] == summary['source_records']
    assert c.sha(render.PROTOCOL) == summary['protocol_sha256']
    assert c.sha(Path(render.__file__)) == summary['source_sha256']
    assert c.sha(root / 'samples.npz') == summary['samples_sha256'] == metrics['samples_sha256']
    labels = np.load(root / 'labels.npy')
    expected_labels = np.load(m.ROOT / model / m.STAGE / 'inputs/labels.npy')
    np.testing.assert_array_equal(labels, expected_labels)
    with np.load(root / 'samples.npz') as data:
        pixels = data['arr_0']
    assert pixels.dtype == np.uint8 and pixels.shape == (400, 256, 256, 3)
    coverage = []
    absolute_difference = 0.
    for record in summary['source_records']:
        path = Path(record['file'])
        assert c.sha(path) == record['sha256']
        with np.load(path) as data:
            start = int(data['start'])
            n = len(data['labels'])
            coverage.extend(range(start, start + n))
            np.testing.assert_array_equal(data['labels'], labels[start:start + n])
            absolute_difference += np.abs(pixels[start:start + n].astype(np.float64) - data['arr_0']).sum()
    assert coverage == list(range(400))
    mae = absolute_difference / pixels.size / 255
    assert abs(mae - summary['mean_absolute_render_difference']) < 1e-10
    features, feature_path = cached_features(root, model)
    assert len(features) == 400 and np.isfinite(features).all()
    with np.load(REFS[model]) as data:
        fid = fid_from_features(features, data['mu'], data['sigma'])
    error = abs(fid - metrics['fid'])
    assert error < 2e-3
    result = dict(passed=True, model=model, track=track, views=400,
                  additional_sampling_paths=0, additional_full_calls=0, additional_prefix_calls=0,
                  source_hashes_verified=True, image_label_coverage_verified=True,
                  samples_sha256=summary['samples_sha256'], source_summary_sha256=c.sha(source),
                  labels_sha256=c.sha(root / 'labels.npy'), features_sha256=c.sha(feature_path),
                  reference_sha256=c.sha(REFS[model]), fid_reported=metrics['fid'],
                  fid_recalculated=fid, fid_absolute_error=error,
                  independent_feature_extraction=False, render_mae_0_1=mae)
    c.atomic(root / 'audit.json', result)
    native = c.read(m.ROOT / model / m.STAGE / (track + '_native') / 'metrics.json')
    fixed = c.read(source.parent / 'metrics.json')
    row = dict(model=model, track=track, native_fid=native['fid'],
               latent_flip_fid=fixed['fid'], pixel_flip_fid=metrics['fid'],
               latent_minus_pixel_fid=fixed['fid'] - metrics['fid'],
               native_is=native['inception_score'], latent_flip_is=fixed['inception_score'],
               pixel_flip_is=metrics['inception_score'], render_mae_0_1=mae,
               extra_decode_seconds=summary['seconds'], additional_sampling_paths=0,
               additional_full_calls=0, additional_prefix_calls=0, diagnostic_views=400)
    return row, result


def main():
    m.configure()
    torch.set_num_threads(4)
    for model in c.MODELS:
        marker = m.ROOT / model / render.STAGE / 'complete.json'
        while not marker.exists():
            time.sleep(5)
        complete = c.read(marker)
        assert complete['complete'] and complete['new_full_calls'] == complete['new_prefix_calls'] == 0
        assert complete['rendered_views'] == 800
    OUT.mkdir(parents=True, exist_ok=True)
    rows, audits = [], []
    for model in c.MODELS:
        for track in ('cfg', 'ig'):
            row, audit = audit_render(model, track)
            rows.append(row)
            audits.append(audit)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'render_comparison.csv', index=False)
    sources = pd.DataFrame([
        dict(source='Render protocol', location=str(render.PROTOCOL), role='Diagnostic specified after primary degradation; no new sampling'),
        dict(source='Primary frozen protocol', location=str(m.PROTOCOL), role='Original 22 arms and unchanged promotion gate'),
        dict(source='Raw source and rendered paths', location=str(m.ROOT), role='Same-path genealogy, image arrays, hashes and cached features'),
        dict(source='Visual Anagrams', location='https://arxiv.org/html/2311.17919v2', role='Prior work explicitly discusses latent transformation artifacts'),
    ])
    with pd.ExcelWriter(OUT / 'source_data.xlsx', engine='openpyxl') as writer:
        for name, table in [('Render comparison', df), ('Sources', sources)]:
            table.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(65, max(18, max(len(str(v.value or '')) for v in column) + 2))
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    x = np.arange(4)
    for offset, name, label, color in [(-.25, 'native_fid', 'Native (different seed orientation)', '#777777'),
                                     (0., 'latent_flip_fid', 'Reflect latent, then decode', '#bc624e'),
                                     (.25, 'pixel_flip_fid', 'Decode, then reflect pixels', '#176b8d')]:
        bars = ax.bar(x + offset, df[name], width=.24, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', fontsize=8)
    ax.set_xticks(x, [NAMES[r.model] + ' ' + r.track.upper() for r in df.itertuples()])
    ax.set_ylabel('FID on 400 images; lower is better')
    ax.set_title('Rendering diagnostic: two operations on the same generated endpoint')
    ax.set_ylim(0, float(df.latent_flip_fid.max()) * 1.25)
    ax.legend(loc='upper center', ncols=3, fontsize=8)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / ('render_comparison.' + ext), dpi=180)
    plt.close(fig)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    for model in c.MODELS:
        canvas = Image.new('RGB', (205 + 192 * 4, 45 + 210 * 4), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), NAMES[model] + ': same paths, first four in order; diagnostic only', font=font, fill='black')
        for i, (track, stage, kind, description) in enumerate([
                ('cfg', m.STAGE, 'cfg_fixed_flip', 'CFG: latent reflection'),
                ('cfg', render.STAGE, 'cfg_physical_flip', 'CFG: pixel reflection'),
                ('ig', m.STAGE, 'ig_fixed_flip', 'IG: latent reflection'),
                ('ig', render.STAGE, 'ig_physical_flip', 'IG: pixel reflection')]):
            root = m.ROOT / model / stage / kind
            with np.load(root / 'samples.npz') as data:
                pixels = data['arr_0'][:4]
            draw.text((10, 70 + i * 210), description, font=font, fill='black')
            draw.text((10, 93 + i * 210), f"FID {c.read(root / 'metrics.json')['fid']:.2f}", font=font, fill='black')
            for j, image in enumerate(pixels):
                canvas.paste(Image.fromarray(image).resize((192, 192)), (205 + j * 192, 45 + i * 210))
        canvas.save(OUT / (model + '_paired_first4.png'))
    columns = ['model', 'track', 'native_fid', 'latent_flip_fid', 'pixel_flip_fid', 'render_mae_0_1']
    text = [
        '# 反射生成失败后的表示检查\n\n',
        '在同一批既有生成端点上，把“latent反射后解码”改成“解码后像素反射”，两模型的CFG与IG质量差距明显收窄。'
        '这支持解码器与latent水平反射不交换，是固定反射臂退化的一个实际来源。它没有证明ABBA交替的全部退化都由解码器造成，'
        '也没有提供新的有效guidance方法。原候选与原晋级规则保持不变。\n\n',
        df[columns].to_markdown(index=False) + '\n\n',
        '![同端点的两种渲染](data/reflection_representation_20260912/render_comparison.png)\n\n',
        '令R为latent水平反射，J为像素水平反射，D为原解码器。固定反射采样的端点是 '
        '$z_R=R\Phi(Rz_0)$，其原图为$D(z_R)$。补充诊断只输出 '
        '$JD(Rz_R)=JD(\Phi(Rz_0))$。两者复用完全相同的采样轨迹，差异在于D和反射操作的次序。\n\n',
        '额外渲染结果从分布上属于反射初始高斯噪声下的原生采样，再做像素反射；高斯噪声分布本来就反射不变。'
        '所以它不是新的guidance候选。表中native来自原始噪声方向，和这两种同端点渲染并非同一条轨迹；'
        '有限样本FID及像素反射对Inception特征的影响也不保证完全消失。不能把小幅优于native的数值解释为方法收益。\n\n',
        '该检查是在SiT固定反射退化后追加，属于事后诊断；不是预先注册的独立确认。两模型各增加800个渲染视图，'
        '合计1600视图，新增采样轨迹、full调用与prefix调用均为0，解码耗时单独保存在数据表。'
        '这些视图和原图相关，不能合并成更大独立样本来评价候选。\n\n',
        '这对加性误差讨论的限制很具体：利用对称性消除预测残差，首先需要选定的变换确实保持模型所处的目标表示。'
        '在像素空间合理的对称性，不能直接假定为每个latent通道的相同空间置换。'
        '否则本来想消除误差的操作自身会增加表示错配。它没有否定“理想分布＋共同错误成分＋结构化错配＋加性残差”的一般结构，'
        '也不识别其中的密度残差或网络误差。\n\n',
        '[Visual Anagrams](https://arxiv.org/html/2311.17919v2)已讨论latent变换产生的伪影，'
        '因此这次发现是本仓库两个具体表示上的实测约束，不作为普遍新原理。'
        '当前固定latent反射构造停止，不继续搜索周期、强度或变换来挽救它。\n\n',
        '![SiT固定前四个](data/reflection_representation_20260912/sit_small_paired_first4.png)\n\n',
        '![RAEv2固定前四个](data/reflection_representation_20260912/raev2_paired_first4.png)\n\n',
        '[原生成结果](REFLECTION_GUIDANCE_RESULTS_20260912_ZH.md) · '
        '[补充协议](REFLECTION_RENDER_AUDIT_20260912_ZH.md) · '
        '[源数据工作簿](data/reflection_representation_20260912/source_data.xlsx) · '
        '[核验记录](data/reflection_representation_20260912/verification.json)。\n',
    ]
    REPORT.write_text(''.join(text))
    workbook = openpyxl.load_workbook(OUT / 'source_data.xlsx', read_only=True)
    assert workbook['Render comparison'].max_row == 5
    workbook.close()
    c.atomic(OUT / 'verification.json', dict(passed=True, diagnostic_arms=4, extra_views=1600,
             new_sampling_paths=0, new_full_calls=0, new_prefix_calls=0, audits=audits,
             max_fid_recalculation_error=max(a['fid_absolute_error'] for a in audits),
             visual_inspection_pending=True, original_selection_unchanged=True))
    c.atomic(OUT / 'artifact_manifest.json', dict(
        files={str(p.relative_to(c.WORK)): c.sha(p) for p in OUT.iterdir() if p.name != 'artifact_manifest.json'},
        report_sha256=c.sha(REPORT), generator_sha256=c.sha(Path(__file__))))
    print('Finalized', REPORT, flush=True)


if __name__ == '__main__':
    main()
