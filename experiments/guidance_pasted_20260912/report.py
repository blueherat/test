"""Refresh the source-backed experiment readout; no research theory expansion."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as c
from experiments import report_replica_reference_20260912 as workbook_writer

OUT=c.WORK/'docs/data/guidance_pasted_20260912'
REPORT=c.WORK/'docs/CFG_IG_PASTED_EXPERIMENTS_20260912_ZH.md'
STAGES=['cfg_screen_400','ig_screen_400','ig_calibrated_screen_400']
NAMES={'independent_base':'普通CFG 原强度','shared_independent_base':'共享CFG 独立噪声 原强度',
    'shared_antithetic_base':'共享CFG 相反噪声 原强度','independent_half':'普通CFG 半强度',
    'shared_independent_half':'共享CFG 独立噪声 半强度','shared_antithetic_half':'共享CFG 相反噪声 半强度',
    'native_base':'原IG','posterior':'来源后验','time_mean':'时间均值','permuted':'同条件置换',
    'reversed':'反向gate','native_half':'原IG 半强度','native_double':'原IG 双强度'}
EN={'independent_base':'CFG base','shared_independent_base':'Shared / independent noise / base',
    'shared_antithetic_base':'Shared / antithetic noise / base','independent_half':'CFG half',
    'shared_independent_half':'Shared / independent noise / half','shared_antithetic_half':'Shared / antithetic noise / half',
    'native_base':'Native IG','posterior':'Source posterior','time_mean':'Time mean','permuted':'Within-class swap',
    'reversed':'Reverse gate','native_half':'Native half','native_double':'Native double'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=Path('/home/zhoushunyu/.codex/attachments/ff9f10a9-fc9c-419f-a385-24f2e29ea24a/pasted-text.txt')
    if source.exists():(OUT/'proposal.txt').write_bytes(source.read_bytes())
    rows=[];blocks=[];calibration=[];training=[]
    for stage in STAGES:
        for model in c.MODELS:
            p=c.ROOT/model/stage/'results.json'
            if not p.exists():continue
            rs=c.read(p)
            blocks += [f'**{model}／{stage}：{len(rs)}组已评价**\n',
                '|方法|FID↓|IS↑|主样本|生成路径|完整前向/输出|采样解码GPU秒|',
                '|---|--:|--:|--:|--:|--:|--:|']
            for r in rs:
                row=dict(model=model,stage=stage,arm=r['arm'],fid=r['fid'],inception_score=r['inception_score'],
                    primary_samples=r['primary_samples'],generated_paths=r['generated_paths'],
                    full_calls_per_output=r['full_calls_per_output'],seconds=r['seconds'])
                rows.append(row)
                blocks.append(f'|{NAMES[r["arm"]]}|{r["fid"]:.4f}|{r["inception_score"]:.4f}|{r["primary_samples"]}|{r["generated_paths"]}|{r["full_calls_per_output"]:.0f}|{r["seconds"]:.2f}|')
            blocks.append('')
    for model in c.MODELS:
        p=c.ROOT/model/'ig_source_data/training_complete.json'
        if p.exists():
            tr=c.read(p);training.append(dict(model=model,parameters=tr['trainable_parameters'],train_seconds=tr['train_seconds'],
                raw_validation_bce=tr['after']['bce'],raw_validation_brier=tr['after']['brier'],raw_validation_accuracy=tr['after']['accuracy']))
        p=c.ROOT/model/'ig_calibrated_source/request.json'
        if p.exists():
            cr=c.read(p);calibration.append(dict(model=model,inverse_temperature=cr['inverse_temperature'],
                raw_audit_bce=cr['audit_raw']['bce'],calibrated_audit_bce=cr['audit_calibrated']['bce'],
                raw_audit_brier=cr['audit_raw']['brier'],calibrated_audit_brier=cr['audit_calibrated']['brier']))
    tables={}
    review=c.read(c.ROOT/'screen_review.json') if (c.ROOT/'screen_review.json').exists() else None
    if review:tables['screen_decisions']=review['decisions']
    for name,rs in [('quality',rows),('training',training),('calibration',calibration)]:
        if not rs:continue
        tables[name]=rs
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    if tables:workbook_writer.OUT=OUT;workbook_writer.workbook(tables)
    fig,axes=plt.subplots(3,2,figsize=(14,12))
    for j,stage in enumerate(STAGES):
        for i,model in enumerate(c.MODELS):
            ax=axes[j,i];rs=[r for r in rows if r['stage']==stage and r['model']==model]
            ax.set_title(f'{model} / '+['Shared CFG','Raw source posterior','Calibrated posterior'][j],fontsize=11)
            if not rs:ax.text(.5,.5,'Evaluation pending',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
            base=(min(r['fid'] for r in rs if r['arm'].startswith('independent_')) if stage=='cfg_screen_400'
                  else next(r['fid'] for r in rs if r['arm']=='native_base'))
            diffs=[r['fid']-base for r in rs]
            ax.barh(range(len(rs)),diffs,color=['#275c83' if r['arm']=='posterior' or r['arm'].startswith('shared_') else '.65' for r in rs])
            ax.set_yticks(range(len(rs)),[EN[r['arm']] for r in rs],fontsize=8);ax.invert_yaxis()
            xlabel='FID minus best ordinary CFG (lower is better)' if stage=='cfg_screen_400' else 'FID minus fixed native IG (lower is better)'
            ax.axvline(0,color='.3',lw=.8);ax.set_xlabel(xlabel,fontsize=9)
            span=max(diffs)-min(diffs);pad=max(3.,span*.35);ax.set_xlim(min(diffs)-pad,max(diffs)+pad)
            for y,(d,r) in enumerate(zip(diffs,rs)):
                ax.annotate(f'{d:+.2f}',(d,y),xytext=(4 if d>=0 else -4,0),textcoords='offset points',va='center',ha='left' if d>=0 else 'right',fontsize=8)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('User-proposed CFG and IG: 400 primary branches, paired inputs within each experiment',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97))
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'quality_comparison.{ext}',dpi=160)
    plt.close(fig)
    statuses={key:c.read(c.ROOT/(key+'.json')) for key in ('cfg_status','ig_status','calibrated_status') if (c.ROOT/(key+'.json')).exists()}
    summary=[]
    for model in c.MODELS:
        rs=[r for r in rows if r['model']==model and r['stage']=='cfg_screen_400']
        if len(rs)==6:
            ordinary=min(r['fid'] for r in rs if r['arm'].startswith('independent_'))
            shared=min(r['fid'] for r in rs if r['arm'].startswith('shared_'))
            summary.append(f'{model}：CFG两点中普通最好{ordinary:.4f}，共享最好{shared:.4f}，差{shared-ordinary:+.4f}。')
    intro='''# 验证用户粘贴的CFG与IG想法：真实生成实验

2026-09-12。按用户要求先验证真实生成效果；保留两条优先候选，局部依赖弱化仍为文本中的备用方向，未当成已经验证的第三项。用户随后追加的[共同污染分布推导](AG_IG_SHARED_CONTAMINATION_DERIVATION_20260912_ZH.md)另文记录，没有改变本轮冻结方法或挑选参数。

CFG使用同条件样本共享额外gap，2×2初始噪声／共享方式，两个固定强度；IG用冻结模型实际Strong/Weak路径训练一个来源分类头，检验来源后验是否值得用于减弱反对Weak的力度。两者都从第一轮纳入SiT与RAEv2。

**评价口径。** 每臂实际生成400对、800条路径，固定第一分支的400张进入主FID/IS；不按图像挑选，不把两条相关输出一起混入主指标。不同实验阶段的噪声不同，不能跨CFG／IG直接比较绝对FID；校准IG刻意复用原IG探索bank，不是独立确认。RAE是预先随机选定的400类子集，参考为完整ImageNet，因此有小样本和类别覆盖限制。这里的IS不等于类别正确率，若有正信号，还需新的完整类别1K／5K与条件正确性检查。

400图仅用于筛选明显成败。时间均值控制由本组原IG轨迹估计固定schedule，它与当前探索bank共用数据，不能把这一控制说成已经完成独立schedule泛化验证；新的确认必须冻结schedule再换噪声。主要后验和置换方法不使用跨样本总体均值。

**实际成本。** CFG每输出SiT224、RAE200次完整前向；IG每输出SiT128、RAE100次。每个独立主评价样本实际用了两条路径的成本，保持两张输出时另有相关性。全部IG推理无独立weak前缀。原IG基线为了构造时间均值还记录r，其计时包含该校准用途；来源头和token池化的开销不能通过NFE抹去。

**一次有依据的修订。** SiT原来源分类头的硬分类准确率高于随机，但NLL与Brier不合格，因此不能叫作已校准后验。保留原实验后，使用来源标签拟合单个逆温度，检查另一半来源轨迹，再运行相同生成对照。校准未使用FID、未更新Strong/Weak或更换MLP；它改善来源概率不代表改善生成。此前已看过200条的汇总验证统计，拆分校准检查仍属于探索性修订。

协议：[CFG共享推动](CFG_SHARED_PUSH_PROTOCOL_20260912_ZH.md) · [IG来源后验](IG_SOURCE_POSTERIOR_PROTOCOL_20260912_ZH.md) · [概率校准修订](IG_SOURCE_CALIBRATION_AMENDMENT_20260912_ZH.md)。原始[用户提案](data/guidance_pasted_20260912/proposal.txt)留档。

只改初始噪声耦合并保持单样本生成映射不变，不会改变单样本边缘；因此独立CFG的相反噪声格只复用已逐位检查相同的第一分支资产，不伪造第二分支。[Couple to Control，§3](https://arxiv.org/html/2605.11311v1)。IG的后验反馈与[Feedback Guidance](https://arxiv.org/html/2506.06085v2)接近，当前不据新公式宣称新颖性。

'''
    conclusion=''
    if review:
        passed=review['eligible']
        conclusion=('**36组固定生成初筛全部完成。** '+('存在通过规则的候选，需独立确认。' if passed else
            'CFG共享推动、原始IG来源后验、校准IG来源后验，在两模型上均未通过预设推进条件。没有追加1K／5K或参数扫描。')+'\n\n')
        conclusion+='|模型|方案|候选FID|规定对照中最好FID|差值|进入独立确认|\n|---|---|--:|--:|--:|---|\n'
        for r in review['decisions']:
            label={'cfg_screen_400':'共享CFG','ig_screen_400':'原始来源IG','ig_calibrated_screen_400':'校准来源IG'}[r['stage']]
            conclusion+=f'|{r["model"]}|{label}|{r["candidate_fid"]:.4f}|{r["baseline_fid"]:.4f}|{r["delta"]:+.4f}|{"是" if r["eligible"] else "否"}|\n'
        conclusion+='\nCFG比较两个强度中最好的普通版本；IG后验须同时胜过原IG、半／双强度、时间均值、同条件置换和反向gate。门槛为FID至少降低2且IS至少保留原对照的90%；这是预先冻结的资源筛选规则，不是显著性检验。\n\n'
        conclusion+='SiT的两项候选均恶化；RAE共享半强度有不足1 FID的微小下降，校准来源IG与原IG基本持平。来源状态的逐样本对应关系也未胜过同条件置换。当前结果不足以支持继续扩大这些固定构造，也不证明所有分布去污染或其他共同污染模型均不可能有效。\n\n'
    text=intro+conclusion+'\n'.join(summary)+'\n\n'+'\n'.join(blocks)
    if calibration:
        text+='\n**来源概率校准（不是图像质量）**\n\n|模型|逆温度|原NLL|校准NLL|原Brier|校准Brier|\n|---|--:|--:|--:|--:|--:|\n'
        for r in calibration:text+=f'|{r["model"]}|{r["inverse_temperature"]:.6f}|{r["raw_audit_bce"]:.4f}|{r["calibrated_audit_bce"]:.4f}|{r["raw_audit_brier"]:.4f}|{r["calibrated_audit_brier"]:.4f}|\n'
    text+='\n![质量结果](data/guidance_pasted_20260912/quality_comparison.png)\n\n[完整数据CSV](data/guidance_pasted_20260912/quality.csv) · [图表数据工作簿](data/guidance_pasted_20260912/source_data.xlsx)。\n'
    if review:
        audit_rows=[]
        for model in c.MODELS:
            for stage in STAGES:
                p=c.ROOT/model/stage/'audit.json'
                if p.exists():audit_rows.extend(c.read(p)['records'])
        worst=max((r['absolute_error'] for r in audit_rows),default=0.)
        text+=f'\n**验证与成本。** 已核对{len(audit_rows)}臂的样本覆盖、输入与产物SHA、标签、潜变量有限性、两条路径以及逐批耗时；用同一缓存特征重算FP64 FID，最大差异{worst:.3g}，未声称使用独立特征提取器。两个模型校准前后的原IG主分支像素完全相同。全部方法源文件与冻结输入检查通过。\n'
        text+=f'\n36臂实际采样与解码共{sum(r["seconds"] for r in rows):.2f}个批次GPU秒，生成{sum(r["generated_paths"] for r in rows):,}条路径；独立主分支评价共{sum(r["primary_samples"] for r in rows):,}张，其中校准原IG是刻意重放。GPU秒是各批设备耗时之和，不是墙钟时间，也不含CPU评价或离线来源生成。\n'
        for model in c.MODELS:
            source=c.read(c.ROOT/model/'ig_source_data/complete.json')
            tr=c.read(c.ROOT/model/'ig_source_data/training_complete.json')
            text+=f'\n{model}：离线来源轨迹{source["source_seconds"]:.2f} GPU秒；{tr["trainable_parameters"]:,}参数的分类头训练{tr["train_seconds"]:.2f} GPU秒。温度校准在CPU完成，未训练去噪模型。\n'
        proof=OUT/'final_provenance.json'
        if proof.exists():
            data=c.read(proof)
            text+='\n来源数据按完整seed分成1000训练对、100校准对和100审计对；所有900份来源特征文件、来源标签、时刻、噪声、温度缩放与原头继承关系已核对。此前已查看过200对的汇总验证，故仍是探索性校准。旧队列及蒸馏STOP文件保持原SHA。[来源与图像归档](data/guidance_pasted_20260912/final_provenance.json)。\n'
            if data['all_six_sample_grids_complete']:
                text+='\n固定前4个主分支图像，无选图：\n\n'
                for model in c.MODELS:
                    links=[f'[{stage}](data/guidance_pasted_20260912/{model}_{stage}_first4.png)' for stage in STAGES]
                    text+=model+'：'+' · '.join(links)+'。\n\n'
        text+='\n本轮两个优先想法的固定验证已结束，备用局部依赖构造未运行；整体研究目标仍未达成。精确去污染系数与本轮有界gate的不同，见[后续推导](AG_IG_SHARED_CONTAMINATION_DERIVATION_20260912_ZH.md)。\n'
    else:
        text+='\n当前执行状态：'+json.dumps({k:v['phase'] for k,v in statuses.items()},ensure_ascii=False)+'。完整验证与结论将在本轮固定队列结束后补齐；当前表格只包括已完成评价的实验。\n'
    REPORT.write_text(text)
    c.atomic(OUT/'artifact_status.json',dict(rows=len(rows),statuses=statuses,proposal_sha256=c.sha(OUT/'proposal.txt')))
    print(json.dumps(dict(rows=len(rows),report=str(REPORT))))


if __name__=='__main__':main()
