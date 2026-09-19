"""Publish only fully audited reference-loss experiments; no sampling mutation."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK=Path(__file__).resolve().parents[1]
BASE=Path('/home/zhoushunyu/data/eqvae/experiments')
OUT=WORK/'docs/data/guidance_reference_loss_20260912'
ROUNDS=(('strong_reference','sit_strong_reference_20260912','strong_reference_screen_1k'),
        ('cfg_posterior','cfg_posterior_reference_20260912','posterior_reference_screen_1k'))
METHODS={'strong_reference':('ig_teacher','ig_strong','ig_teacherstrong','cfg_strong'),
         'cfg_posterior':('cfg_posterior','cfg_independent')}


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_sheets():
    selections=(('ig','sit_strong_reference_20260912','strong_reference_screen_1k',
        [('ig_original_00','Original IG'),('ig_real_00','Real / FM'),('ig_teacher_00','Real / teacher'),
         ('ig_strong_00','Strong samples / FM'),('ig_ig_00','IG samples / FM'),('ig_adg_reference_00','ADG')]),
        ('cfg','cfg_posterior_reference_20260912','posterior_reference_screen_1k',
        [('cfg_original_00','Original CFG'),('cfg_real_00','Real / FM'),('cfg_posterior_00','Paired teacher'),
         ('cfg_independent_00','Independent teacher'),('cfg_cfg_00','CFG samples / FM'),('cfg_apg_07','APG')]))
    for name,folder,stage,methods in selections:
        fig,axes=plt.subplots(4,len(methods),figsize=(12,8),layout='constrained')
        labels=None
        for j,(arm,title) in enumerate(methods):
            path=BASE/folder/stage/arm/'rank0/batch0000.npz'
            with np.load(path) as batch:
                pictures=batch['arr_0'];ys=batch['labels']
                assert pictures.shape==(8,256,256,3) and pictures.dtype==np.uint8
                if labels is None:labels=ys.copy()
                else:np.testing.assert_array_equal(labels,ys)
            for i in range(4):
                axes[i,j].imshow(pictures[i]);axes[i,j].axis('off')
                if i==0:axes[i,j].set_title(title,fontsize=10)
        fig.suptitle('First four paired inputs, in original order; no sample selection',fontsize=13)
        fig.savefig(OUT/f'{name}_first4.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={};all_rows=[];by_round={};training=[];reviews={}
    for name,folder,stage in ROUNDS:
        root=BASE/folder;base=root/stage
        status=read(root/'status.json');assert status['phase']=='complete' and status['final_audit_passed']
        audit_path=WORK/'docs/data'/stage/'audit.json';audit=read(audit_path)
        request=read(base/'request.json');rows=read(base/'results.json')
        assert audit['passed'] and len(audit['raw_and_metric_audits'])==len(rows)==len(request['configs'])
        assert all(a['raw_batch_and_aggregate_hashes_verified'] and a['costs_reconciled'] for a in audit['raw_and_metric_audits'])
        assert audit['request_sha256']==sha(base/'request.json')
        for p in (root/'status.json',base/'request.json',base/'results.json',audit_path,root/'screen_review.json'):
            manifest[str(p)]=sha(p)
        reviews[name]=read(root/'screen_review.json')
        by_arm={r['arm']:r for r in rows};by_round[name]=by_arm
        for r in rows:
            assert r['complete']
            control=r['source']+'_real_00';fid_control=by_arm[control]['fid']
            record=dict(round=name,arm=r['arm'],source=r['source'],role=r['role'],fid=r['fid'],
                sfid=r['metrics']['sfid'],inception_score=r['metrics']['inception_score'],
                full_calls=r['full_calls_per_image'],prefix_calls=r['prefix_calls_per_image'],
                sampling_decode_gpu_seconds=r['sum_batch_gpu_seconds'],matched_real=control,
                delta_fid_vs_real=r['fid']-fid_control,request_sha256=sha(base/'request.json'))
            all_rows.append(record)
        for method in METHODS[name]:
            p=root/'training'/method/'complete.json';r=read(p);manifest[str(p)]=sha(p)
            assert r['passed'] and r['strong_unchanged'] and r['strong_gradients_absent']
            training.append(dict(round=name,method=method,parameters=r['trainable_parameters'],
                training_seconds=r['training_seconds'],guided_mse_before=r['validation_before']['guided_mse'],
                guided_mse_after=r['validation_after']['guided_mse'],checkpoint_sha256=r['checkpoint_sha256']))
    with (OUT/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    with (OUT/'training.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(training[0]));w.writeheader();w.writerows(training)
    panels=(('IG: same real-data training control',
        [('strong_reference','ig_teacher_00','Real data / teacher target'),
         ('strong_reference','ig_strong_00','Strong samples / FM target'),
         ('strong_reference','ig_teacherstrong_00','Strong samples / teacher target'),
         ('strong_reference','ig_ig_00','IG samples / FM (previous construction)')]),
        ('CFG: each round uses its matching real-data control',
        [('strong_reference','cfg_strong_00','Strong samples / FM target'),
         ('cfg_posterior','cfg_posterior_00','Paired conditional teacher'),
         ('cfg_posterior','cfg_independent_00','Independent-label teacher (control)'),
         ('cfg_posterior','cfg_cfg_00','CFG samples / FM (previous construction)')]))
    fig,axes=plt.subplots(2,1,figsize=(10.4,6.9),layout='constrained')
    for ax,(title,choices) in zip(axes,panels):
        values=[];labels=[]
        for round_name,arm,label in choices:
            r=by_round[round_name][arm];control=by_round[round_name][r['source']+'_real_00']
            values.append(r['fid']-control['fid']);labels.append(label)
        ax.barh(range(len(values)),values,color=['#28778f' if v<0 else '#b97158' for v in values])
        ax.set_yticks(range(len(values)),labels,fontsize=9);ax.invert_yaxis()
        ax.axvline(0,color='#777777',lw=.8);ax.axvline(-.5,color='#777777',lw=1,ls='--')
        span=max(.8,max(abs(v) for v in values))
        ax.set_xlim(min(-.7,min(values)-span*.12),max(.45,max(values)+span*.16))
        for i,v in enumerate(values):ax.text(v+(.015*span if v>=0 else -.015*span),i,f'{v:+.3f}',
            ha='right' if v<0 else 'left',va='center',fontsize=9)
        ax.set_title(title,fontsize=11);ax.set_xlabel('FID difference; lower is better')
        ax.grid(axis='x',alpha=.12)
    fig.suptitle('Reference-loss experiments: exploratory paired 1K, no uncertainty intervals',fontsize=12)
    for suffix in ('png','pdf','svg'):fig.savefig(OUT/f'quality_comparison.{suffix}',dpi=190)
    plt.close(fig)
    sample_sheets()
    g=by_round['strong_reference'];p=by_round['cfg_posterior']
    lines=['新增六个读出训练、两轮共21组配对1K已全部完成，原始batch与缓存FID/sFID详细审计均通过。全部冻结配置均未改变引导窗口、步数或训练预算。',
        '', '|IG：第三轮|FID↓|相对同预算real/FM|', '|---|--:|--:|']
    ig_labels=(('ig_original_00','原IG'),('ig_real_00','real/FM'),('ig_teacher_00','real/teacher'),
        ('ig_strong_00','strong samples/FM'),('ig_teacherstrong_00','strong samples/teacher'),
        ('ig_ig_00','IG自身samples/FM'),('ig_half_00','固定半强度原IG'),('ig_adg_reference_00','ADG'))
    for arm,label in ig_labels:
        r=g[arm];lines.append(f'|{label}|{r["fid"]:.4f}|{r["fid"]-g["ig_real_00"]["fid"]:+.4f}|')
    lines += ['',f'第三轮CFG strong-data为{g["cfg_strong_00"]["fid"]:.4f}，同预算real/FM为{g["cfg_real_00"]["fid"]:.4f}，CFG自身数据为{g["cfg_cfg_00"]["fid"]:.4f}，APG为{g["cfg_apg_07"]["fid"]:.4f}。',
        '', '|纯CFG：第四轮|FID↓|sFID↓|IS↑|', '|---|--:|--:|--:|']
    cfg_labels=(('cfg_original_00','原CFG'),('cfg_real_00','real/FM'),('cfg_posterior_00','配对conditional teacher'),
        ('cfg_independent_00','独立类别teacher，对照'),('cfg_cfg_00','CFG自身samples/FM'),
        ('cfg_half_00','固定半强度原CFG'),('cfg_apg_07','APG'))
    for arm,label in cfg_labels:
        r=p[arm];lines.append(f'|{label}|{r["fid"]:.4f}|{r["metrics"]["sfid"]:.4f}|{r["metrics"]["inception_score"]:.4f}|')
    lines += ['', '![固定loss的质量结果](data/guidance_reference_loss_20260912/quality_comparison.png)', '',
        '图中每一差值都使用同轮重跑的real/FM对照；虚线只表示对这个对照的.5推进门槛，正式推进还要求同时超过所有指定强对照。无置信区间，不能用短线长短宣称显著性。', '',
        '|新增训练|参数|训练GPU秒|留出guided MSE|', '|---|--:|--:|--:|']
    for r in training:lines.append(f'|{r["method"]}|{r["parameters"]}|{r["training_seconds"]:.2f}|{r["guided_mse_after"]:.7f}|')
    total_training=sum(r['training_seconds'] for r in training)
    total_sampling=sum(r['sampling_decode_gpu_seconds'] for r in all_rows)
    data_path=BASE/'sit_strong_reference_20260912/data/complete.json'
    generation=read(data_path)['generation_gpu_seconds']['strong'];manifest[str(data_path)]=sha(data_path)
    lines += ['',f'本次新增strong训练样本合成{generation:.2f} GPU秒，六个新头训练合计{total_training:.2f} GPU秒，21组质量采样/解码合计{total_sampling:.2f} GPU秒。均为各batch/作业的GPU占用时段之和，不是多卡墙钟加速比；不含模型加载、预检、FID评估和审计。', '',
        f'第三轮推进名单：{reviews["strong_reference"]["eligible_for_fixed_5k"]}；第四轮：{reviews["cfg_posterior"]["eligible_for_fixed_5k"]}。规则及SHA在各轮screen_review.json保留。', '',
        '[全部21组FID/sFID/IS与实际成本](data/guidance_reference_loss_20260912/results.csv) · [六个训练记录](data/guidance_reference_loss_20260912/training.csv) · [第三轮详细审计](data/strong_reference_screen_1k/audit.json) · [第四轮详细审计](data/posterior_reference_screen_1k/audit.json)']
    lines += ['', '固定展示第一个batch的前4个输入，不筛选样本；少量图像不承担质量优劣结论。', '',
        '![IG固定配对样本](data/guidance_reference_loss_20260912/ig_first4.png)', '',
        '![CFG固定配对样本](data/guidance_reference_loss_20260912/cfg_first4.png)']
    body='\n'.join(lines)
    report=WORK/'docs/GUIDANCE_REFERENCE_LOSS_RESEARCH_20260912_ZH.md'
    text=report.read_text()
    placeholder='第三、四轮的结果将在全部生成和原始文件审计完成后补入本节。未完成的结果不提前算作成功。各轮的1K都来自已经反复探索的同一噪声bank；.5 FID是预先规定的推进门槛，不是统计显著性标准。只有达到全部强对照门槛才冻结权重进入新5K，不因为 MSE 降低就扩大训练或改变引导强度。'
    start='<!-- EXPERIMENT_RESULTS_START -->';end='<!-- EXPERIMENT_RESULTS_END -->'
    if start in text:
        a,b=text.index(start),text.index(end)+len(end);text=text[:a]+start+'\n'+body+'\n'+end+text[b:]
    else:
        assert placeholder in text;text=text.replace(placeholder,start+'\n'+body+'\n'+end)
    text=text.replace('第三轮已训练四个新读出、正在完成14组配对1K；第四轮纯CFG的两个loss及七组对照已固定，接续执行。',
        '第三轮四个新读出与14组配对1K、第四轮纯CFG两个新读出与7组配对1K均已完成并通过全部详细审计。')
    report.write_text(text)
    (OUT/'artifact_manifest.json').write_text(json.dumps(dict(inputs=manifest,
        source_sha256=sha(Path(__file__)),all_rounds_complete_and_audited=True,rows=len(all_rows),
        training_runs=len(training),promotion=reviews),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(rows=len(all_rows),training_runs=len(training),total_sampling_decode_gpu_seconds=total_sampling,
        total_training_gpu_seconds=total_training,promoted={k:v['eligible_for_fixed_5k'] for k,v in reviews.items()})))


if __name__=='__main__':main()
