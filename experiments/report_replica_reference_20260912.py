"""Source-backed report for the frozen paired-noise IG experiment."""
from pathlib import Path
import csv
import hashlib
import json
import re
import zipfile
from xml.sax.saxutils import escape
from xml.etree import ElementTree
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import report_compiled_guidance_20260912 as sheets

WORK = Path(__file__).resolve().parents[1]
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/sit_replica_reference_20260912')
OUT = WORK/'docs/data/replica_reference_20260912'
REPORT = WORK/'docs/CFG_IG_REPLICA_RESEARCH_20260912_ZH.md'
STAGE = 'replica_screen_1k'
METHODS = ('ig_pair_mse', 'ig_pair_consistent', 'ig_pair_average', 'ig_pair_shuffled')
NAMES = dict(zip([m+'_00' for m in METHODS]+['ig_original_00','ig_ig_00','ig_adg_reference_00'],
    ['同预算MSE','同图一致性（主要候选）','先平均（反向对照）','打乱图像配对','原IG','已有自身数据参考','ADG']))
EN = dict(zip(NAMES, ['Matched MSE','Same-image consistency','Prediction averaging',
    'Shuffled image pairs','Native IG','Prior self-data reference','ADG']))


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def workbook(tables):
    """Small dependency-free OOXML workbook containing the exact chart data."""
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    def column(j):
        s = ''
        while j:
            j, k = divmod(j-1, 26); s = chr(65+k)+s
        return s
    files = {}
    overrides = '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    sheet_names = []; relationships = []
    for idx, (name, rows) in enumerate(tables.items(), 1):
        keys = list(rows[0]); grid = [keys]+[[r[k] for k in keys] for r in rows]; xml_rows = []
        for i, row in enumerate(grid, 1):
            cells = []
            for j, v in enumerate(row, 1):
                ref = f'{column(j)}{i}'
                if isinstance(v, (float, int)) and not isinstance(v, bool):
                    cells.append(f'<c r="{ref}"><v>{v}</v></c>')
                else:
                    cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>')
            xml_rows.append(f'<row r="{i}">'+''.join(cells)+'</row>')
        files[f'xl/worksheets/sheet{idx}.xml'] = f'<worksheet xmlns="{ns}"><sheetData>'+''.join(xml_rows)+'</sheetData></worksheet>'
        sheet_names.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
        relationships.append(f'<Relationship Id="rId{idx}" Type="{rel}/worksheet" Target="worksheets/sheet{idx}.xml"/>')
        overrides += f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    files['[Content_Types].xml'] = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'+overrides+'</Types>'
    files['_rels/.rels'] = f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    files['xl/workbook.xml'] = f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>'+''.join(sheet_names)+'</sheets></workbook>'
    files['xl/_rels/workbook.xml.rels'] = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(relationships)+'</Relationships>'
    with zipfile.ZipFile(OUT/'source_data.xlsx', 'w', zipfile.ZIP_DEFLATED) as z:
        for name, xml in files.items():
            ElementTree.fromstring(xml)
            z.writestr(name, '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+xml)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = read(ROOT/'status.json'); assert status['phase'] == 'complete' and status['final_audit_passed']
    request = read(ROOT/STAGE/'request.json'); results = read(ROOT/STAGE/'results.json')
    audit_path = WORK/'docs/data'/STAGE/'audit.json'; audit = read(audit_path)
    assert audit['passed'] and audit['results'] == 7 and audit['request_sha256'] == sha(ROOT/STAGE/'request.json')
    inputs = {str(p): sha(p) for p in [ROOT/'status.json', ROOT/STAGE/'request.json', ROOT/STAGE/'results.json', audit_path]}
    rows = []
    for r in results:
        assert r['complete'] and r['full_calls_per_image'] == 128 and r['prefix_calls_per_image'] == 0
        rows.append(dict(arm=r['arm'], fid=r['fid'], sfid=r['metrics']['sfid'], inception_score=r['metrics']['inception_score'],
            full_calls=128, prefix_calls=0, sample_decode_seconds=r['sum_batch_gpu_seconds']))
    by = {r['arm']:r for r in rows}; write_csv(OUT/'results.csv', rows)
    training = []
    for method in METHODS:
        p = ROOT/'training'/method/'complete.json'; r = read(p); inputs[str(p)] = sha(p)
        tr = dict(method=method, parameters=r['trainable_parameters'], seconds=r['training_seconds'],
            initial_head_sha256=r['initial_head_sha256'], input_fingerprints_sha256=r['input_fingerprints_sha256'])
        for k in ('mse','paired_prediction_disagreement','paired_average_mse'):
            tr['before_'+k] = r['validation_before'][k]; tr['after_'+k] = r['validation_after'][k]
        training.append(tr)
    assert len({r['initial_head_sha256'] for r in training}) == 1
    assert len({r['input_fingerprints_sha256'] for r in training}) == 1
    write_csv(OUT/'training.csv', training)
    curves = list(csv.DictReader((OUT/'two_point_curves.csv').open()))
    curves = [{k:float(v) for k,v in r.items()} for r in curves]
    theory = read(OUT/'theory_checks.json'); assert theory['passed']
    numeric = [{'check':k,'value':v} for k,v in theory.items() if isinstance(v,(int,float,bool))]
    for r in theory['gaussian_corruption_curl_counterexample']:
        numeric.extend({'check':f'quadrature_{r["quadrature_order"]}_{k}','value':v} for k,v in r.items() if k != 'quadrature_order')
    workbook(dict(quality=rows, training=training, two_point=curves, theory=numeric))

    fig, axes = plt.subplots(1,3,figsize=(12,3.8), sharey=True)
    for ax, sigma in zip(axes, sorted({r['sigma'] for r in curves})):
        c = [r for r in curves if r['sigma'] == sigma]; x = [r['y'] for r in c]
        for key,label,color in [('mse','Strong = exact posterior mean','.25'),('consistent_guided','Subtract consistency reference','#275c83'),('average_guided','Subtract averaging reference','#b55b24')]:
            ax.plot(x,[r[key] for r in c],label=label,color=color,lw=1.6)
        ax.axhline(1,color='.7',ls=':',lw=.8); ax.axhline(-1,color='.7',ls=':',lw=.8)
        ax.set_title(f'Two-point prior, sigma = {sigma}',fontsize=11)
        ax.set_xlabel('Noisy observation y')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Predicted clean value')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,fontsize=9)
    fig.suptitle('Opposite risk changes have opposite actions; neither is universal density smoothing',fontsize=11)
    fig.subplots_adjust(left=.065,right=.99,bottom=.23,top=.79,wspace=.20)
    for ext in ('png','pdf','svg'): fig.savefig(OUT/f'two_point_actions.{ext}',dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.5,4.3))
    base = by['ig_pair_mse_00']['fid']; diffs = [r['fid']-base for r in rows]
    ax.barh(range(len(rows)),diffs,color=['#275c83' if r['arm']=='ig_pair_consistent_00' else '#b55b24' if r['arm']=='ig_pair_average_00' else '.65' for r in rows])
    ax.set_yticks(range(len(rows)),[EN[r['arm']] for r in rows]); ax.invert_yaxis(); ax.axvline(0,color='.3',lw=.8)
    for i,(r,d) in enumerate(zip(rows,diffs)):
        ax.annotate(f'{d:+.2f}  (FID {r["fid"]:.2f})',(d,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=9)
    ax.set_xlim(min(diffs)-.6,max(diffs)+4.5)
    ax.set(xlabel='FID difference from matched MSE reference (lower is better)',title='Frozen paired-noise IG experiment: seven paired 1K arms')
    ax.spines[['top','right']].set_visible(False); fig.tight_layout()
    for ext in ('png','pdf','svg'): fig.savefig(OUT/f'quality_comparison.{ext}',dpi=170)
    plt.close(fig)
    sheets.ROOT,sheets.OUT,sheets.EN = ROOT,OUT,EN
    sheets.samples(STAGE,results,'ig')

    train_cost = sum(r['seconds'] for r in training); sample_cost = sum(r['sample_decode_seconds'] for r in rows)
    blocks = ['**七组新配对1000图：相同噪声、标签及采样器**\n','|方法|FID↓|sFID↓|IS↑|Full/prefix|采样与解码GPU秒|','|---|--:|--:|--:|--:|--:|']
    blocks += [f'|{NAMES[r["arm"]]}|{r["fid"]:.4f}|{r["sfid"]:.4f}|{r["inception_score"]:.4f}|128/0|{r["sample_decode_seconds"]:.2f}|' for r in rows]
    blocks += ['','|训练|参数|训练GPU秒|保留集MSE|同图预测分歧|平均预测MSE|','|---|--:|--:|--:|--:|--:|']
    blocks += [f'|{NAMES[r["method"]+"_00"]}|{r["parameters"]}|{r["seconds"]:.2f}|{r["after_mse"]:.6f}|{r["after_paired_prediction_disagreement"]:.6f}|{r["after_paired_average_mse"]:.6f}|' for r in training]
    delta = by['ig_pair_consistent_00']['fid']-base
    blocks += ['',f'主要候选相对同预算MSE的FID差为 **{delta:+.4f}**。一致性损失降低了同图分歧，先平均损失降低了平均预测误差，但二者均明显损害生成。打乱配对接近MSE，不支持把结果解释为多取一次噪声或简单缩放训练loss即可改善。',
        f'四个训练合计{train_cost:.2f} GPU秒；七组采样/解码合计{sample_cost:.2f} GPU秒。时间不含模型加载、数据读取、特征提取及审计，不能由前向次数相同推断延迟严格相同。共同初始化与原始候选随机流一致；打乱组按设计使用第二张图，因此实际监督输入并非四组全部相同。',
        '原始batch、样本覆盖、权重和成本检查通过；FID/sFID由同一缓存特征另用FP64算术核对，这不是独立特征提取验证。1K用于筛除明显失败候选，不能支持小幅改善的统计结论。固定推进名单为空，未启动5K、跨模型或系数搜索。',
        '[逐组质量与成本](data/replica_reference_20260912/results.csv) · [训练记录](data/replica_reference_20260912/training.csv) · [图表数据工作簿](data/replica_reference_20260912/source_data.xlsx)',
        '![质量比较](data/replica_reference_20260912/quality_comparison.png)',
        '固定展示首批前4个输入，无图像筛选；这些少量样本不承担总体结论。',
        f'![配对样本](data/replica_reference_20260912/{STAGE}_ig_first4.png)']
    summary = '四个真实数据IG读出和七组新配对1K均已完成，审计通过。主要候选FID74.27，同预算MSE65.50，已有参考64.47；本轮没有生成收益。配对loss分支停止，未启动5K、迁移或系数搜索。'
    text = REPORT.read_text()
    text = re.sub(r'<!-- STATUS_START -->.*?<!-- STATUS_END -->','<!-- STATUS_START -->\n'+summary+'\n<!-- STATUS_END -->',text,flags=re.S)
    text = re.sub(r'<!-- RESULTS_START -->.*?<!-- RESULTS_END -->','<!-- RESULTS_START -->\n'+'\n\n'.join(blocks)+'\n<!-- RESULTS_END -->',text,flags=re.S)
    # Keep Markdown tables contiguous while preserving paragraph separation.
    text = re.sub(r'\|\n\n\|','|\n|',text)
    REPORT.write_text(text)
    manifest = dict(passed=True, inputs=inputs, status=status, rows=len(rows), training_seconds=train_cost,
        sample_decode_seconds=sample_cost, candidate_minus_matched_mse_fid=delta, workbook_sheets=4)
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='inputs'}))


if __name__ == '__main__': main()
