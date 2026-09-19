from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/sit_reference_compilation_20260912')
OUT=WORK/'docs/data/reference_compilation_20260912'
REPORT=WORK/'docs/CFG_IG_COMPILED_RESEARCH_20260912_ZH.md'
NAMES={'ig_original_00':'原IG','ig_shallow_00':'浅层参考蒸馏','ig_deep_00':'末层参考蒸馏',
    'ig_ig_00':'既有自身数据参考','ig_adg_reference_00':'ADG','cfg_original_00':'原CFG',
    'cfg_probability_00':'候选概率组合','cfg_mean_00':'同读出均值外推','cfg_cfg_00':'既有自身数据参考','cfg_apg_07':'APG'}
EN={'ig_original_00':'Native IG','ig_shallow_00':'Shallow distilled reference','ig_deep_00':'Deep distilled reference',
    'ig_ig_00':'Prior self-data reference','ig_adg_reference_00':'ADG','cfg_original_00':'Native CFG',
    'cfg_probability_00':'Candidate probability update','cfg_mean_00':'Same-head mean extrapolation',
    'cfg_cfg_00':'Prior self-data reference','cfg_apg_07':'APG'}


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def samples(stage,rows,source):
    arms=[r['arm'] for r in rows if r['source']==source]
    fig,axes=plt.subplots(len(arms),4,figsize=(8,2*len(arms)))
    expected=None
    for i,arm in enumerate(arms):
        with np.load(ROOT/stage/arm/'rank0/batch0000.npz') as record:
            imgs=record['arr_0'][:4];labels=record['labels'][:4]
            if expected is None:expected=labels.copy()
            else:np.testing.assert_array_equal(labels,expected)
        for j,img in enumerate(imgs):
            axes[i,j].imshow(img);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if j==0:axes[i,j].set_ylabel(EN[arm],fontsize=8)
            if i==0:axes[i,j].set_title(f'Input {j}, class {labels[j]}',fontsize=9)
    fig.suptitle(f'{source.upper()}: first four paired inputs, no selection',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/f'{stage}_{source}_first4.png',dpi=150);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);status=read(ROOT/'status.json')
    assert status['phase']=='complete' and status['final_audit_passed']
    stages=['compiled_screen_1k']
    if (ROOT/'compiled_confirm_5k/results.json').exists():stages.append('compiled_confirm_5k')
    portable=[];inputs={};blocks=[]
    for stage in stages:
        request=read(ROOT/stage/'request.json');rows=read(ROOT/stage/'results.json')
        audit=WORK/'docs/data'/stage/'audit.json';assert audit.exists()
        for path in (ROOT/stage/'request.json',ROOT/stage/'results.json',audit):inputs[str(path)]=sha(path)
        blocks.extend([f'**{request["samples"]}图配对评估**\n',
            '|主线／方法|FID↓|sFID↓|IS↑|Full/prefix|采样与解码GPU秒|','|---|--:|--:|--:|--:|--:|'])
        for r in rows:
            assert r['complete'] and r['prefix_calls_per_image']==0
            blocks.append(f'|{r["source"].upper()}：{NAMES[r["arm"]]}|{r["fid"]:.4f}|{r["metrics"]["sfid"]:.4f}|'
                f'{r["metrics"]["inception_score"]:.4f}|{r["full_calls_per_image"]:.0f}/0|{r["sum_batch_gpu_seconds"]:.2f}|')
            portable.append(dict(stage=stage,arm=r['arm'],source=r['source'],role=r['role'],fid=r['fid'],
                sfid=r['metrics']['sfid'],inception_score=r['metrics']['inception_score'],
                full_calls=r['full_calls_per_image'],prefix_calls=r['prefix_calls_per_image'],
                sample_decode_seconds=r['sum_batch_gpu_seconds'],request_sha256=sha(ROOT/stage/'request.json')))
        for source in ('ig','cfg'):
            if any(r['source']==source for r in rows):samples(stage,rows,source)
        blocks.append('')
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(portable[0]));writer.writeheader();writer.writerows(portable)
    training=[]
    for method in ('ig_shallow','ig_deep','cfg_categorical'):
        path=ROOT/'training'/method/'complete.json';r=read(path);inputs[str(path)]=sha(path)
        training.append(dict(method=method,parameters=r['trainable_parameters'],seconds=r['training_seconds'],
            validation=json.dumps(r['validation_after'])))
    with (OUT/'training.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(training[0]));w.writeheader();w.writerows(training)
    data_seconds=sum(read(ROOT/'data'/f'rollout_rank{rank}.json')['seconds'] for rank in range(4))
    codebook_seconds=read(ROOT/'data/codebook.json')['seconds']
    training_seconds=sum(r['seconds'] for r in training);sample_seconds=sum(r['sample_decode_seconds'] for r in portable)
    blocks.extend(['**离线训练与推理成本**\n','|训练|参数|GPU秒|','|---|--:|--:|'])
    blocks.extend(f'|{r["method"]}|{r["parameters"]}|{r["seconds"]:.2f}|' for r in training)
    blocks.extend(['',f'离线教师轨迹生成{data_seconds:.2f} GPU秒，codebook Lloyd更新{codebook_seconds:.2f} GPU秒，三个读出训练合计{training_seconds:.2f} GPU秒。'
        f'本轮{sum(read(ROOT/stage/"request.json")["samples"]*len(read(ROOT/stage/"results.json")) for stage in stages)}张质量图像的采样/解码合计{sample_seconds:.2f} GPU秒。'
        '这些是同步后的作业或batch时段之和，不是多卡墙钟加速比；不含模型加载、缓存读取、FID提取和审计，codebook时间不含训练patch抽样。\n',
        f'预设1K推进名单：{status["eligible"]}；5K确认名单：{status["confirmed"]}。\n',
        '[逐组质量与实际成本](data/reference_compilation_20260912/results.csv) · [训练记录](data/reference_compilation_20260912/training.csv)\n',
        '![配对质量比较](data/reference_compilation_20260912/quality_comparison.png)\n',
        '以下固定展示第一批前4个输入，无图像筛选；少量样本不承担整体质量结论。\n',
        '![IG配对样本](data/reference_compilation_20260912/compiled_screen_1k_ig_first4.png)\n',
        '![CFG配对样本](data/reference_compilation_20260912/compiled_screen_1k_cfg_first4.png)\n'])
    screen=[r for r in portable if r['stage']=='compiled_screen_1k'];fig,axes=plt.subplots(1,2,figsize=(13,4.2))
    for ax,source in zip(axes,('ig','cfg')):
        values=[r for r in screen if r['source']==source];base=next(r['fid'] for r in values if r['arm']==source+'_original_00')
        delta=[r['fid']-base for r in values];positions=np.arange(len(values))
        ax.barh(positions,delta,color=['#275c83' if r['role']=='candidate' else '.65' for r in values])
        ax.set_yticks(positions,[EN[r['arm']] for r in values],fontsize=8);ax.invert_yaxis();ax.axvline(0,color='.2',lw=.8)
        ax.set(title=f'{source.upper()} paired 1K',xlabel='FID difference from same-bank native (lower is better)')
        for y,d,r in zip(positions,delta,values):
            ax.annotate(f'{d:+.2f}  (FID {r["fid"]:.2f})',(d,y),xytext=(4 if d>=0 else -4,0),
                textcoords='offset points',ha='left' if d>=0 else 'right',va='center',fontsize=8)
        lo,hi=min(delta),max(delta);pad=max(1.,(hi-lo)*.5);ax.set_xlim(lo-pad,hi+pad)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'quality_comparison.{ext}',dpi=180)
    plt.close(fig)
    summary=f'三个读出与10组新配对1K已全部完成，详细核验通过。'
    if status['eligible']:summary+=f'进入固定5K的候选为{status["eligible"]}，本轮确认名单为{status["confirmed"]}。'
    else:summary+='没有候选通过预设推进门槛，未启动5K或迁移。'
    text=REPORT.read_text();text=re.sub(r'<!-- STATUS_START -->.*?<!-- STATUS_END -->',
        '<!-- STATUS_START -->\n'+summary+'\n<!-- STATUS_END -->',text,flags=re.S)
    text=re.sub(r'<!-- RESULTS_START -->.*?<!-- RESULTS_END -->','<!-- RESULTS_START -->\n'+'\n'.join(blocks)+'\n<!-- RESULTS_END -->',text,flags=re.S)
    REPORT.write_text(text)
    manifest=dict(passed=True,inputs=inputs,status=status,rows=len(portable),teacher_data_seconds=data_seconds,
        codebook_lloyd_seconds=codebook_seconds,training_seconds=training_seconds,sample_decode_seconds=sample_seconds)
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest))


if __name__=='__main__':main()
