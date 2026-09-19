from pathlib import Path
import csv
import hashlib
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import report_compiled_guidance_20260912 as sheets

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/sit_bayes_risk_reference_20260912')
OUT=WORK/'docs/data/bayes_risk_reference_20260912'
REPORT=WORK/'docs/IG_BAYES_RISK_RESEARCH_20260912_ZH.md'
NAMES={'ig_square_00':'同预算MSE','ig_quartic_00':'四次损失','ig_original_00':'原IG','ig_ig_00':'已有自身数据参考','ig_adg_reference_00':'ADG'}
EN={'ig_square_00':'Matched squared loss','ig_quartic_00':'Quartic loss','ig_original_00':'Native IG','ig_ig_00':'Prior self-data reference','ig_adg_reference_00':'ADG'}


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True);status=read(ROOT/'status.json')
    assert status['phase']=='complete' and status['final_audit_passed']
    stages=['risk_screen_1k']
    if (ROOT/'risk_confirm_5k/results.json').exists():stages.append('risk_confirm_5k')
    rows_out=[];blocks=[];inputs={}
    for stage in stages:
        request=read(ROOT/stage/'request.json');rows=read(ROOT/stage/'results.json')
        audit_path=WORK/'docs/data'/stage/'audit.json';audit=read(audit_path)
        assert audit['passed'] and audit['results']==len(rows) and audit['request_sha256']==sha(ROOT/stage/'request.json')
        for p in (ROOT/stage/'request.json',ROOT/stage/'results.json',audit_path):inputs[str(p)]=sha(p)
        blocks += [f'**新配对{request["samples"]}图**\n','|方法|FID↓|sFID↓|IS↑|Full/prefix|采样与解码GPU秒|','|---|--:|--:|--:|--:|--:|']
        for r in rows:
            assert r['complete'] and r['full_calls_per_image']==128 and r['prefix_calls_per_image']==0
            blocks.append(f'|{NAMES[r["arm"]]}|{r["fid"]:.4f}|{r["metrics"]["sfid"]:.4f}|{r["metrics"]["inception_score"]:.4f}|128/0|{r["sum_batch_gpu_seconds"]:.2f}|')
            rows_out.append(dict(stage=stage,arm=r['arm'],fid=r['fid'],sfid=r['metrics']['sfid'],inception_score=r['metrics']['inception_score'],
                full_calls=r['full_calls_per_image'],prefix_calls=r['prefix_calls_per_image'],sample_decode_seconds=r['sum_batch_gpu_seconds']))
    with (OUT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0]));w.writeheader();w.writerows(rows_out)
    training=[]
    for method in ('ig_square','ig_quartic'):
        path=ROOT/'training'/method/'complete.json';r=read(path);inputs[str(path)]=sha(path)
        training.append(dict(method=method,parameters=r['trainable_parameters'],seconds=r['training_seconds'],
            initial_head_sha256=r['initial_head_sha256'],before_mse=r['validation_before']['mse'],after_mse=r['validation_after']['mse'],
            before_fourth=r['validation_before']['fourth_moment'],after_fourth=r['validation_after']['fourth_moment']))
    assert training[0]['initial_head_sha256']==training[1]['initial_head_sha256']
    with (OUT/'training.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(training[0]));w.writeheader();w.writerows(training)
    blocks += ['','|训练目标|参数|训练GPU秒|保留集MSE|保留集四阶误差|','|---|--:|--:|--:|--:|']
    blocks += [f'|{r["method"]}|{r["parameters"]}|{r["seconds"]:.2f}|{r["after_mse"]:.6f}|{r["after_fourth"]:.6f}|' for r in training]
    train_cost=sum(r['seconds'] for r in training);sample_cost=sum(r['sample_decode_seconds'] for r in rows_out)
    screen=[r for r in rows_out if r['stage']=='risk_screen_1k'];by={r['arm']:r for r in screen}
    delta=by['ig_quartic_00']['fid']-by['ig_square_00']['fid']
    blocks += ['',f'共同初始化的保留集MSE为{training[0]["before_mse"]:.6f}、四阶误差为{training[0]["before_fourth"]:.6f}。'
        f'两个训练合计{train_cost:.2f} GPU秒，全部质量采样/解码合计{sample_cost:.2f} GPU秒。时间不含加载、数据读取、特征提取与审计；不把相同前向次数说成延迟完全相同。\n',
        f'四次损失相对同预算MSE的FID差为{delta:+.4f}。推进名单{status["eligible"]}，确认名单{status["confirmed"]}。\n',
        '[逐组质量与成本](data/bayes_risk_reference_20260912/results.csv) · [训练与保留集记录](data/bayes_risk_reference_20260912/training.csv)\n',
        '![配对质量比较](data/bayes_risk_reference_20260912/quality_comparison.png)\n',
        '固定展示首批前4个输入，无图像筛选；不由这4张图推断总体效果。\n',
        '![配对样本](data/bayes_risk_reference_20260912/risk_screen_1k_ig_first4.png)\n']
    fig,ax=plt.subplots(figsize=(9.2,3.7));baseline=by['ig_original_00']['fid']
    diffs=[r['fid']-baseline for r in screen]
    ax.barh(range(len(screen)),diffs,color=['#275c83' if r['arm']=='ig_quartic_00' else '.65' for r in screen])
    ax.set_yticks(range(len(screen)),[EN[r['arm']] for r in screen]);ax.invert_yaxis();ax.axvline(0,color='.2',lw=.8)
    ax.set(xlabel='FID difference from same-bank native (lower is better)',title='Squared versus quartic reference: paired 1K')
    for i,(d,r) in enumerate(zip(diffs,screen)):
        ax.annotate(f'{d:+.2f}  (FID {r["fid"]:.2f})',(d,i),xytext=(5 if d>=0 else -5,0),textcoords='offset points',va='center',ha='left' if d>=0 else 'right',fontsize=9)
    ax.set_xlim(min(diffs)-1.3,max(diffs)+1.3);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'quality_comparison.{ext}',dpi=170)
    plt.close(fig)
    sheets.ROOT,sheets.OUT,sheets.EN=ROOT,OUT,EN
    sheets.samples('risk_screen_1k',read(ROOT/'risk_screen_1k/results.json'),'ig')
    summary='两个真实数据读出和五组新配对1K均已完成，原始样本与缓存指标核验通过。'
    summary+=('固定5K已执行，确认名单为'+str(status['confirmed'])+'。') if status['eligible'] else '四次损失未通过推进门槛，未启动5K、迁移或损失指数网格。'
    text=REPORT.read_text();text=re.sub(r'<!-- STATUS_START -->.*?<!-- STATUS_END -->','<!-- STATUS_START -->\n'+summary+'\n<!-- STATUS_END -->',text,flags=re.S)
    text=re.sub(r'<!-- RESULTS_START -->.*?<!-- RESULTS_END -->','<!-- RESULTS_START -->\n'+'\n'.join(blocks)+'\n<!-- RESULTS_END -->',text,flags=re.S)
    REPORT.write_text(text)
    manifest=dict(passed=True,inputs=inputs,status=status,training_seconds=train_cost,sample_decode_seconds=sample_cost,rows=len(rows_out),quartic_minus_square_fid=delta)
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))


if __name__=='__main__':main()
