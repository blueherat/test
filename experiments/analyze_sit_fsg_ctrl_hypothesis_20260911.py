"""Inspectable paired summaries and unselected image comparisons for the study."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from experiments.sit_fsg_ctrl_hypothesis_20260911 import pipeline as p,study
from experiments.lifting_scale_sweep_20260909 import atomic,read,sha

OUT=p.PORTABLE
FIG=OUT/'figures'
NAMES=dict(cfg_tuned='CFG a=1.25',cfg_high='CFG a=2.75',fsg_high='FSG SiT adapter',
    fsg_length_high='FSG length / CFG direction',fsg_debiased_high='FSG minus same-field error',
    cycle_high='Same-field Euler cycle',smc_high='CTRL K=0.2',instant_high='Instant K=0.2',
    instant01_high='Instant K=0.1',soft_high='Soft threshold',norm_high='Scalar norm control',
    refined_high='Heun refinement',local_fit='Local-gap fit',agreement_fit='Agreement fit',
    write_fit='Fixed conditional-target fit',guided_write_fit='Fixed guided-target fit')
COLORS=dict(cfg_tuned='#277DA1',cfg_high='#5B6470',fsg_high='#D17A22',smc_high='#8059A3',
    instant_high='#287D64',fsg_length_high='#CE5163',soft_high='#AD8650',norm_high='#8A9A5B',
    refined_high='#569CA4')
MAIN=('cfg_tuned','cfg_high','fsg_high','smc_high','instant_high')


def load(part,key='rows'):
    values=[]
    request_hash=sha(p.ROOT/'study_request.json')
    for path in sorted((p.ROOT/part).glob('*.json')):
        if path.name.startswith('rank'):continue
        item=read(path)
        assert item['request_sha256']==request_hash,path
        values.extend(item.get(key,[]))
    return pd.DataFrame(values)


def grouped(frame,keys):
    numeric=[c for c in frame.select_dtypes(include=np.number).columns if c not in keys+['index','start']]
    values=frame.groupby(keys,dropna=False)[numeric].mean().reset_index()
    values['n']=frame.groupby(keys,dropna=False).size().to_numpy()
    return values


def paired_interventions(frame):
    keys=['index','base','k']
    baseline=frame[frame.variant=='unchanged'].set_index(keys)
    metrics=['write_rms','agreement_rms','local_gap_rms','guided_write_rms','conditional_drift_rms',
        'u_resnet_p','u_resnet_top1','u_convnext_p','u_convnext_top1',
        'c_resnet_p','c_resnet_top1','c_convnext_p','c_convnext_top1']
    joined=frame.join(baseline[metrics],on=keys,rsuffix='_before',validate='many_to_one')
    for key in metrics:
        joined['delta_'+key]=joined[key]-joined[key+'_before']
        if key.endswith('_rms') and key!='conditional_drift_rms':
            joined['reduction_'+key]=1-joined[key]/joined[key+'_before'].clip(lower=1e-10)
    return joined


def paired_trajectory(frame):
    keys=['index','k','suffix']
    metrics=['resnet_p','resnet_top1','convnext_p','convnext_top1']
    rows=[]
    for control in ('cfg_tuned','cfg_high','smc_high'):
        baseline=frame[frame.method==control].set_index(keys)
        joined=frame.join(baseline[metrics],on=keys,rsuffix='_before',validate='many_to_one')
        joined['control']=control
        for key in metrics:
            joined['delta_'+key]=joined[key]-joined[key+'_before']
        rows.append(joined)
    return pd.concat(rows,ignore_index=True)


def bootstrap_mean(values,seed=202612121):
    values=np.asarray(values,dtype=float)
    if len(values)<2:return (float(values.mean()),float(values.mean()))
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
    return tuple(np.quantile(means,[.025,.975]).tolist())


def pair_intervals(frame,keys,metrics):
    rows=[]
    for context,group in frame.groupby(keys):
        row=dict(zip(keys,context if isinstance(context,tuple) else (context,)))
        for key in metrics:
            lo,hi=bootstrap_mean(group[key])
            row.update({key:float(group[key].mean()),key+'_lo':lo,key+'_hi':hi})
        row['n']=len(group);rows.append(row)
    return pd.DataFrame(rows)


def gap_readout_correlations(frame):
    """Within-method/time associations; class differences are not controlled away."""
    values=[]
    for (method,k),part in frame[frame.suffix=='null'].groupby(['method','k']):
        for metric in ('convnext_p','resnet_p'):
            result=spearmanr(part.gap_rms,part[metric])
            values.append(dict(method=method,k=k,readout=metric,n=len(part),
                rho=float(result.statistic),p_value_descriptive=float(result.pvalue)))
    pd.DataFrame(values).to_csv(OUT/'gap_readout_correlations.csv',index=False)


def settings():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':160,
        'figure.facecolor':'white','axes.facecolor':'white','grid.alpha':.18})


def savefig(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def trajectory_figures(frame,traces):
    settings();means=grouped(frame,['method','k','suffix'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,classifier in zip(axes,('resnet','convnext')):
        for method in MAIN:
            rows=means[(means.method==method)&(means.suffix=='null')].sort_values('k')
            ax.plot(rows.k/64,100*rows[classifier+'_top1'],marker='o',ms=4,color=COLORS[method],label=NAMES[method])
        ax.set(xlabel='Time at which all conditioning is removed',title=classifier.capitalize(),ylim=(0,100))
        ax.grid(True);ax.set_xticks(np.array(study.TIMES)/64)
    axes[0].set_ylabel('Target class top-1 after NULL suffix (%)')
    axes[1].legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Fixed 200-image bank; every method uses the same noise and labels',y=1.01)
    savefig(fig,'null_handoff_alignment')
    trace=grouped(traces,['method','k'])
    fig,axes=plt.subplots(1,3,figsize=(14,3.8))
    for method in MAIN:
        rows=trace[trace.method==method]
        for ax,key in zip(axes,('gap_rms','clean_gap_rms','noise_gap_rms')):
            ax.plot(rows.k/64,rows[key],color=COLORS[method],label=NAMES[method])
            ax.set(xlabel='Time',ylabel='RMS',title=key.replace('_',' '));ax.grid(True)
    axes[-1].legend(frameon=False,fontsize=8)
    savefig(fig,'gap_parameterizations')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for method in ('smc_high','instant_high','soft_high','norm_high'):
        rows=trace[trace.method==method]
        axes[0].plot(rows.k/64,100*rows.sign_disagreement,label=NAMES[method],color=COLORS[method])
        axes[1].plot(rows.k/64,100*rows.guidance_flip_fraction,label=NAMES[method],color=COLORS[method])
    axes[0].set(title='History changes the correction sign',ylabel='Coordinates (%)',xlabel='Time')
    axes[1].set(title='Modified guidance reverses raw gap',ylabel='Coordinates (%)',xlabel='Time')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(True)
    savefig(fig,'ctrl_coordinate_traces')


def jacobian_figures(frame):
    settings()
    actual=frame[~frame.variant.isin(['resolution','short_limit'])]
    # Average over the predeclared 2 strengths, 3 times and 32 shared samples.
    means=grouped(actual,['horizon','variant'])
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    specifications=[
        ('finite_target_relative_error',('flow_inverse','flow_cfg_equal'),'Fine forward / inverse flow'),
        ('finite_target_relative_error',('euler_fsg','euler_cfg_equal','euler_debiased'),'Euler FSG adapter'),
        ('best_linear_ray_relative_error',('euler_capped','capped_cfg_equal','flow_capped'),'Direction after optimal linear scaling'),
        ('direction_cos_cfg',('euler_fsg','flow_inverse','euler_debiased'),'State displacement vs local CFG')]
    names=dict(flow_inverse='Flow inverse',flow_cfg_equal='CFG, same length',euler_fsg='Euler FSG',
        euler_cfg_equal='CFG, same length',euler_debiased='Euler minus cycle',euler_capped='Capped Euler FSG',
        capped_cfg_equal='CFG, same capped length',flow_capped='Capped flow inverse')
    for ax,(metric,variants,title) in zip(axes.flat,specifications):
        for variant in variants:
            rows=means[means.variant==variant].sort_values('horizon')
            ax.plot(rows.horizon,rows[metric],marker='o',label=names[variant])
        ax.set(title=title,xlabel='Horizon H',ylabel='Relative error' if metric!='direction_cos_cfg' else 'Cosine')
        ax.set_xticks([2/64,4/64,8/64,16/64],['2/64','4/64','8/64','16/64'])
        ax.grid(True);ax.legend(frameon=False,fontsize=8)
        if metric=='finite_target_relative_error':ax.set_yscale('log')
    fig.suptitle('All target errors use the fine guided-flow endpoint; Euler-own target is audited separately',fontsize=12)
    fig.tight_layout();savefig(fig,'jacobian_flow_response')


def jacobian_audit_summaries():
    audit=p.ROOT/'independent_jacobian_audit'
    result=read(audit/'results.json')
    assert result['passed'] and result['request_sha256']==sha(audit/'request.json')
    ad=pd.DataFrame(result['rows']);ad.to_csv(OUT/'independent_jvp_audit.csv',index=False)
    atomic(OUT/'independent_jvp_cpu_audit.json',result['cpu'])
    own=p.ROOT/'own_target_audit'
    if not (own/'complete.json').exists():return
    complete=read(own/'complete.json');assert complete['complete'] and complete['rows']==5376
    request_hash=sha(own/'request.json');assert complete['request_sha256']==request_hash
    rows=[]
    for rank in range(4):
        result=read(own/f'rank{rank}.json')
        assert result['complete'] and result['request_sha256']==request_hash
        rows+=result['rows']
    frame=pd.DataFrame(rows)
    assert len(frame)==5376 and not frame.duplicated(['index','base','k','horizon','variant']).any()
    frame.to_csv(OUT/'own_target_rows.csv',index=False)
    grouped(frame,['horizon','variant']).to_csv(OUT/'own_target_means.csv',index=False)
    grouped(frame,['amount','horizon','variant']).to_csv(OUT/'own_target_means_by_amount.csv',index=False)
    keys=['index','base','k','horizon'];pairs=[]
    for capped,left,right in [(False,'euler_fsg','cfg_equal'),(True,'euler_capped','cfg_capped')]:
        a=frame[frame.variant==left].set_index(keys)
        b=frame[frame.variant==right].set_index(keys)
        for metric in ('finite_error_flow_target','finite_error_own_euler_target'):
            delta=(a[metric]-b[metric]).rename('fsg_minus_cfg').reset_index()
            # Repeated times and strengths remain in the same independent-noise unit.
            units=delta.groupby(['index','horizon']).fsg_minus_cfg.mean().reset_index()
            for horizon,part in units.groupby('horizon'):
                low,high=bootstrap_mean(part.fsg_minus_cfg)
                pairs.append(dict(capped=capped,target=metric,horizon=horizon,n_independent_noises=len(part),
                    mean_fsg_minus_cfg=float(part.fsg_minus_cfg.mean()),bootstrap95_low=low,bootstrap95_high=high))
    pd.DataFrame(pairs).to_csv(OUT/'own_target_paired_intervals.csv',index=False)
    settings();means=grouped(frame,['horizon','variant']);fig,axes=plt.subplots(2,2,figsize=(11,8))
    names=dict(euler_fsg='Euler FSG',cfg_equal='CFG, same length',euler_capped='Capped Euler FSG',
        cfg_capped='CFG, same capped length',flow_inverse='Fine inverse of this target',
        own_euler_inverse='Fine inverse of this target')
    for col,(metric,title,oracle) in enumerate([
        ('finite_error_flow_target','Fine guided-flow target','flow_inverse'),
        ('finite_error_own_euler_target','Adapter own Euler-forward target','own_euler_inverse')]):
        for row,variants in enumerate([('euler_fsg','cfg_equal',oracle),('euler_capped','cfg_capped')]):
            ax=axes[row,col]
            for variant,color in zip(variants,('#D17A22','#5B6470','#277DA1')):
                selected=means[means.variant==variant].sort_values('horizon')
                ax.plot(selected.horizon,selected[metric],marker='o',color=color,label=names[variant])
            ax.set(xlabel='Horizon H',ylabel='Finite endpoint relative error',title=title+(' · uncapped' if row==0 else ' · capped'))
            ax.set_xticks([2/64,4/64,8/64,16/64],['2/64','4/64','8/64','16/64'])
            ax.grid(True);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Same 32 noises × 2 strengths × 3 times; direct endpoint errors without a Jacobian approximation',fontsize=12)
    fig.tight_layout();savefig(fig,'fsg_two_targets_audit')


def intervention_figures(frame):
    settings()
    selected=['cfg_radius','fsg_radius','local_radius','agreement_radius','write_radius','guided_write_radius']
    means=grouped(frame[frame.variant.isin(selected)],['variant'])
    means=means.set_index('variant').loc[selected]
    labels=['CFG','FSG','Local gap','Agreement','Fixed C target','Fixed guided target']
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for ax,key,title in zip(axes,['reduction_write_rms','reduction_agreement_rms','delta_u_convnext_top1'],
                          ['Fixed-target error reduction','Moving-target error reduction','NULL target top-1 change']):
        ax.barh(np.arange(len(means)),100*means[key],color=['#5B6470','#D17A22','#8B8DAB','#BC6270','#39846E','#277DA1'])
        ax.set_yticks(np.arange(len(means)),labels);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.8)
        ax.set(title=title,xlabel='Percent' if key.startswith('reduction') else 'Percentage points')
        ax.grid(True,axis='x')
    fig.suptitle('Equal displacement radius; 64 shared states × 2 strengths × 3 times',y=1.01)
    fig.tight_layout();savefig(fig,'fixed_radius_interventions')


def class_names():
    records=read(p.semantic.MANIFEST)['classes']
    return {int(r['label']):r['name'].split(',')[0] for r in records}


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def comparison_sheet(frame,suffix='guided',step=64,indices=(0,1,2,3)):
    names=class_names();labels=np.load(p.ROOT/'inputs/labels.npy')
    tile=256;pad=12;left=220;top=62;caption=42
    sheet=Image.new('RGB',(left+len(MAIN)*(tile+pad)+pad,top+len(indices)*(tile+caption+pad)+pad),'white')
    draw=ImageDraw.Draw(sheet)
    for j,method in enumerate(MAIN):
        draw.text((left+j*(tile+pad),15),NAMES[method],font=font(19),fill='#17212b')
    for i,index in enumerate(indices):
        y0=top+i*(tile+caption+pad)
        label=names[int(labels[index])]
        draw.text((12,y0+20),f'#{index:03d}',font=font(22),fill='#17212b')
        words=label.split();lines=['']
        for word in words:
            if len(lines[-1]+' '+word)>17:lines.append(word)
            else:lines[-1]=(lines[-1]+' '+word).strip()
        for n,line in enumerate(lines):draw.text((12,y0+58+n*24),line,font=font(18),fill='#17212b')
        for j,method in enumerate(MAIN):
            folder='guided' if suffix=='guided' else f'{suffix}_{step:02d}'
            path=p.ROOT/'images'/'trajectories'/method/folder/f'{index:03d}.png'
            if not path.exists():continue
            image=Image.open(path).convert('RGB');assert image.size==(256,256)
            x0=left+j*(tile+pad);sheet.paste(image,(x0,y0))
            row=frame[(frame['index']==index)&(frame.method==method)&(frame.k==step)&(frame.suffix==suffix)]
            if len(row):
                r=row.iloc[0]
                draw.text((x0,y0+tile+5),f'P(target): R18 {r.resnet_p:.3f} / CN {r.convnext_p:.3f}',font=font(13),fill='#334155')
    FIG.mkdir(parents=True,exist_ok=True)
    path=FIG/f'examples_{suffix}_{step:02d}.png';sheet.save(path)
    return path


def intervention_sheet(frame,base='cfg_tuned',step=24,indices=(0,1,2,3)):
    """Unselected first four samples; pixel tiles are copied without resampling."""
    columns=[('reference_c','Frozen C(x)'),('unchanged','Original NULL'),
        ('cfg_radius','CFG, fixed radius'),('fsg_radius','FSG, fixed radius'),
        ('agreement_radius','Agreement direction'),('write_radius','Fixed-target direction'),
        ('full_inverse_oracle','Full inverse oracle')]
    names=class_names();labels=np.load(p.ROOT/'inputs/labels.npy')
    tile=256;pad=12;left=220;top=90;row_height=322
    sheet=Image.new('RGB',(left+len(columns)*(tile+pad)+pad,top+len(indices)*row_height+pad),'white')
    draw=ImageDraw.Draw(sheet)
    draw.text((12,8),f'Same-state intervention · {NAMES[base]} · t={step/64:g} · first four fixed samples',font=font(24),fill='#17212b')
    for j,(_,label) in enumerate(columns):draw.text((left+j*(tile+pad),51),label,font=font(17),fill='#17212b')
    root=p.ROOT/'images/interventions'/base/f'{step:02d}'
    for i,index in enumerate(indices):
        y0=top+i*row_height
        draw.text((12,y0+20),f'#{index:03d}',font=font(22),fill='#17212b')
        words=names[int(labels[index])].split();lines=['']
        for word in words:
            if len(lines[-1]+' '+word)>17:lines.append(word)
            else:lines[-1]=(lines[-1]+' '+word).strip()
        for line,text in enumerate(lines):draw.text((12,y0+58+24*line),text,font=font(18),fill='#17212b')
        for j,(variant,_) in enumerate(columns):
            path=root/variant/(f'{index:03d}.png' if variant=='reference_c' else f'u/{index:03d}.png')
            pixel=Image.open(path).convert('RGB');assert pixel.size==(256,256)
            x0=left+j*(tile+pad);sheet.paste(pixel,(x0,y0))
            lookup='unchanged' if variant=='reference_c' else variant
            record=frame[(frame.base==base)&(frame.k==step)&(frame['index']==index)&(frame.variant==lookup)].iloc[0]
            probability=record.reference_c_convnext_p if variant=='reference_c' else record.u_convnext_p
            draw.text((x0,y0+261),f'CN P(target) {probability:.3f}',font=font(14),fill='#334155')
            if variant!='reference_c':
                draw.text((x0,y0+282),f'write {record.write_rms:.4f} · shift {record.shift_rms:.4f}',font=font(14),fill='#334155')
    path=FIG/f'intervention_examples_{base}_{step:02d}.png';sheet.save(path)
    return path


def gallery(frame,interventions):
    labels=np.load(p.ROOT/'inputs/labels.npy');names=class_names()
    samples=[dict(index=i,label=names[int(label)]) for i,label in enumerate(labels)]
    quality={f'{r.method}/{r.suffix}/{r.k}/{r.index}':dict(r18=r.resnet_p,cn=r.convnext_p,
        r18ok=r.resnet_top1,cnok=r.convnext_top1) for r in frame.itertuples(index=False)}
    intervention_records=interventions.replace({np.nan:None}).to_dict('records') if len(interventions) else []
    data=json.dumps(dict(samples=samples,methods=study.TRAJECTORIES,names=NAMES,quality=quality,
        interventions=intervention_records),ensure_ascii=False).replace('</','<\\/')
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SiT FSG / CTRL 图片检查</title><style>
body{font:15px system-ui,sans-serif;color:#17212b;background:#f4f6f8;margin:24px}h1{font-size:25px}header,nav{max-width:1300px}nav{display:flex;gap:16px;flex-wrap:wrap;margin:22px 0}label{display:grid;gap:5px}select{padding:9px;border:1px solid #bbc5d0;border-radius:6px;background:white}table{border-collapse:separate;border-spacing:10px}th{text-align:left;vertical-align:top;min-width:120px}td{background:white;border:1px solid #d5dde4;border-radius:5px;padding:9px;vertical-align:top}img{width:224px;height:224px;display:block;object-fit:contain}small{display:block;line-height:1.5;color:#425466}.scroll{overflow-x:auto}a{color:#176994}p{line-height:1.65}.muted{color:#526270}
</style><header><h1>FSG / CFG-Ctrl：同噪声、同目标的真实生成图片</h1>
<p>机制集为独立的200张，100类各2张。所有样本都可查看；初始展示前4张的静态比较没有按效果挑图。R18 / CN 为两个后验分类器的目标概率，不能代替画质判断。</p>
<p class="muted">时间从噪声0到图像1。NULL后缀撤掉全部类别条件；纯条件后缀保留类别输入，只移除额外CFG。点击图片查看原始256×256像素。</p></header>
<nav><label>检查内容<select id="mode"><option value="trajectory">完整轨迹与撤条件后缀</option><option value="intervention">同状态单次干预</option></select></label><label>样本 / 目标<select id="sample"></select></label><label>后缀切换时刻<select id="tailstep"><option value="8">t=0.125</option><option value="16">t=0.250</option><option value="24">t=0.375</option><option value="32" selected>t=0.500</option><option value="40">t=0.625</option><option value="48">t=0.750</option></select></label><label>干预前缀<select id="base"><option value="cfg_tuned">CFG a=1.25</option><option value="cfg_high">CFG a=2.75</option></select></label><label>干预时刻<select id="step"><option value="24">t=0.375</option><option value="8">t=0.125</option><option value="40">t=0.625</option></select></label></nav>
<div id="note"></div><div class="scroll" id="table"></div><script>const D=__DATA__;
const sample=document.querySelector('#sample'),mode=document.querySelector('#mode'),base=document.querySelector('#base'),step=document.querySelector('#step'),tailstep=document.querySelector('#tailstep');
D.samples.forEach(s=>{let o=document.createElement('option');o.value=s.index;o.textContent=`#${String(s.index).padStart(3,'0')} · ${s.label}`;sample.append(o)});
const img=(path,caption)=>`<a href="${path}" target="_blank"><img loading="lazy" src="${path}" alt="配对生成图片"></a><small>${caption}</small>`;
function draw(){let i=+sample.value,pad=String(i).padStart(3,'0'),body='',head='';base.disabled=step.disabled=mode.value!=='intervention';tailstep.disabled=mode.value!=='trajectory';
if(mode.value==='trajectory'){
let k=+tailstep.value,when=(k/64).toFixed(3),columns=[['guided',64,'原 guidance 完成'],['null',k,`t=${when} → NULL`],['conditional',k,`t=${when} → 纯条件`]];
head='<tr><th>方法</th>'+columns.map(c=>`<th>${c[2]}</th>`).join('')+'</tr>';
for(let m of D.methods){body+=`<tr><th>${D.names[m]}</th>`;for(let [suffix,k] of columns){let q=D.quality[`${m}/${suffix}/${k}/${i}`],folder=suffix==='guided'?'guided':`${suffix}_${String(k).padStart(2,'0')}`;body+='<td>'+(q?img(`images/trajectories/${m}/${folder}/${pad}.png`,`R18 ${q.r18.toFixed(3)} · CN ${q.cn.toFixed(3)}<br>top1: ${q.r18ok?'✓':'×'} / ${q.cnok?'✓':'×'}`):'尚未生成')+'</td>'}body+='</tr>'}
document.querySelector('#note').textContent='同一列表示相同的后缀切换时刻；完整计算成本见报告。';
}else{let rows=D.interventions.filter(r=>r.index===i&&r.base===base.value&&r.k===+step.value);
head='<tr><th>单次更新</th><th>修改后 NULL 结局</th><th>修改后条件结局</th><th>实际变化</th></tr>';
for(let r of rows){let dir=`images/interventions/${r.base}/${String(r.k).padStart(2,'0')}/${r.variant}`;body+=`<tr><th>${r.variant}</th><td>${img(`${dir}/u/${pad}.png`,`R18 ${r.u_resnet_p.toFixed(3)} · CN ${r.u_convnext_p.toFixed(3)}`)}</td><td>${img(`${dir}/c/${pad}.png`,`R18 ${r.c_resnet_p.toFixed(3)} · CN ${r.c_convnext_p.toFixed(3)}`)}</td><td><small>位移RMS ${r.shift_rms.toFixed(4)}<br>冻结目标误差 ${r.write_rms.toFixed(4)}<br>移动目标误差 ${r.agreement_rms.toFixed(4)}<br>局部gap ${r.local_gap_rms.toFixed(4)}</small></td></tr>`}
if(rows.length){let ref=rows[0],dir=`images/interventions/${ref.base}/${String(ref.k).padStart(2,'0')}`;document.querySelector('#note').innerHTML='<p>前64张有干预；完整反演oracle仅前32张。以下两个参考目标在修改前冻结。4维拟合方向共用同一子空间，oracle的幅度与成本另计。</p><div style="display:flex;gap:24px;flex-wrap:wrap"><div>'+img(`${dir}/reference_c/${pad}.png`,`冻结原始条件未来 C(x) · CN ${ref.reference_c_convnext_p.toFixed(3)}`)+'</div><div>'+img(`${dir}/reference_guided/${pad}.png`,`冻结引导未来 G(x) · CN ${ref.reference_guided_convnext_p.toFixed(3)}`)+'</div></div>'}else{document.querySelector('#note').textContent='这个样本不在前64张干预子集中。'}}
document.querySelector('#table').innerHTML='<table>'+head+body+'</table>'}
for(let e of [sample,mode,base,step,tailstep])e.addEventListener('change',draw);draw();</script></html>'''
    (p.ROOT/'gallery.html').write_text(document.replace('__DATA__',data))


def quality_records():
    values=[]
    request=read(p.ROOT/p.STAGE/'request.json')
    for config in request['configs']:
        directory=p.ROOT/p.STAGE/config['arm']
        if not (directory/'commit.json').exists():continue
        commit=read(directory/'commit.json')
        assert commit['request_sha256']==sha(p.ROOT/p.STAGE/'request.json')
        assert sha(directory/'result.json')==commit['files']['result.json']
        result=read(directory/'result.json')
        values.append(dict(result,method=config['parameters']['method'],
            handoff=config['parameters'].get('handoff'),tail=config['parameters'].get('tail'),
            directory=str(directory),reused=False))
    for method,arm in p.REUSED.items():
        directory=p.SOURCE/arm
        result=read(directory/'result.json')
        commit=read(directory/'commit.json')
        assert sha(directory/'result.json')==commit['files']['result.json']
        assert commit['request_sha256']==sha(p.SOURCE/'request.json')
        values.append(dict(result,arm=method,source_arm=arm,method=method,handoff=None,tail=None,
            directory=str(directory),reused=True))
    return values


def quality_summary():
    rows=quality_records();final=[]
    labels=np.load(p.ROOT/p.STAGE/'inputs/labels.npy')
    classifier_assets=None
    for item in rows:
        row={k:item.get(k) for k in ['arm','source_arm','method','handoff','tail','reused','fid','complete',
            'full_calls_per_image','prefix_calls_per_image','sum_batch_gpu_seconds','directory']}
        if item['complete']:
            row.update({k:item['metrics'][k] for k in ('sfid','inception_score')})
            alignment=OUT/'quality_alignment'/f'{item["arm"]}.npz'
            if alignment.exists():
                if classifier_assets is None:
                    classifier_assets={str(x):sha(x) for x in p.semantic.source_assets()}
                receipt=read(alignment.with_suffix('.json'))
                assert receipt['sha256']==sha(alignment)
                assert receipt['identity']==dict(classifier_assets=classifier_assets,
                    method='Both classifiers read the exact FID uint8 pixels, full ImageNet-1K targets',
                    labels_sha256=p.array_sha(labels),arm=item['arm'],samples_sha256=item['sample_sha256'])
                with np.load(alignment) as data:
                    np.testing.assert_array_equal(data['labels'],labels)
                    for key in ('resnet_p','resnet_top1','convnext_p','convnext_top1'):
                        assert data[key].shape==(1000,) and np.isfinite(data[key]).all()
                        assert ((data[key]>=0)&(data[key]<=1)).all()
                        row[key]=float(data[key].mean())
            activation_path=Path(item['directory'])/'activations.npz'
            assert sha(activation_path)==item['activations_sha256']
            with np.load(activation_path) as data:
                key='pool_3' if 'pool_3' in data.files else ('pool' if 'pool' in data.files else data.files[0])
                feature=data[key].astype(np.float64)
                assert feature.shape==(1000,2048),(key,feature.shape)
            centered=feature-feature.mean(0)
            row['inception_total_variance']=float(np.square(centered).sum()/999)
            within=0.
            for label in range(100):
                subset=feature[labels==label]
                assert len(subset)==10
                within+=np.square(subset-subset.mean(0)).sum()
            row['inception_within_target_variance']=float(within/900)
        final.append(row)
    frame=pd.DataFrame(final)
    if len(frame):
        reference=frame[frame.arm=='cfg_tuned'].iloc[0]
        frame['fid_delta_tuned_cfg']=frame.fid-reference.fid
        frame['sampling_cost_ratio_tuned_cfg']=frame.sum_batch_gpu_seconds/reference.sum_batch_gpu_seconds
        frame['within_target_variance_ratio_tuned_cfg']=frame.inception_within_target_variance/reference.inception_within_target_variance
        frame.to_csv(OUT/'quality_all_results.csv',index=False)
    return frame


def score_quality(rank=None):
    """Posthoc pixel readouts, launched after the sampling GPUs are released."""
    import torch
    from types import SimpleNamespace
    rows=quality_records()
    assert len(rows)==36 and all(row['complete'] for row in rows)
    if rank is not None:rows=[row for i,row in enumerate(rows) if i%4==rank]
    reader=study.Readouts(SimpleNamespace())
    folder=OUT/'quality_alignment';folder.mkdir(parents=True,exist_ok=True)
    labels=np.load(p.ROOT/p.STAGE/'inputs/labels.npy')
    identity=dict(classifier_assets={str(x):sha(x) for x in p.semantic.source_assets()},
        method='Both classifiers read the exact FID uint8 pixels, full ImageNet-1K targets',
        labels_sha256=p.array_sha(labels))
    with torch.inference_mode():
        for row in rows:
            path=folder/f'{row["arm"]}.npz';receipt=path.with_suffix('.json')
            image_path=Path(row['directory'])/'samples.npz'
            expected=dict(identity,arm=row['arm'],samples_sha256=sha(image_path))
            if path.exists() and receipt.exists():
                got=read(receipt)
                assert got['identity']==expected and got['sha256']==sha(path)
                continue
            with np.load(image_path) as data:pixels=data['arr_0']
            assert pixels.shape==(1000,256,256,3) and pixels.dtype==np.uint8
            values={}
            for start in range(0,1000,8):
                y=torch.from_numpy(labels[start:start+8].copy()).cuda()
                output=reader.images(pixels[start:start+8],y)
                for key,value in output.items():values.setdefault(key,[]).append(value)
            values={key:np.concatenate(value) for key,value in values.items()}
            np.savez(path,labels=labels,**values)
            atomic(receipt,dict(identity=expected,sha256=sha(path)))
            print(dict(scored=row['arm'],resnet_top1=float(values['resnet_top1'].mean()),
                convnext_top1=float(values['convnext_top1'].mean())),flush=True)
    if rank is None:quality_summary()


def score_quality_parallel():
    import os
    import subprocess
    import sys
    import time
    assert len(quality_records())==36
    folder=OUT/'quality_alignment';folder.mkdir(parents=True,exist_ok=True)
    workers=[];streams=[]
    try:
        for rank in range(4):
            stream=(folder/f'rank{rank}.log').open('a');streams.append(stream)
            workers.append(subprocess.Popen([sys.executable,'-u','-m',
                'experiments.analyze_sit_fsg_ctrl_hypothesis_20260911','--score-quality','--rank',str(rank)],
                cwd=p.WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                stdout=stream,stderr=subprocess.STDOUT))
        while any(worker.poll() is None for worker in workers):
            assert all(worker.poll() in (None,0) for worker in workers),'Quality classifier worker failed'
            time.sleep(1)
        assert all(worker.returncode==0 for worker in workers)
    finally:
        for worker in workers:
            if worker.poll() is None:worker.terminate()
        for worker in workers:worker.wait(timeout=30)
        for stream in streams:stream.close()
    quality_summary()


def quality_figures(frame):
    settings()
    full=frame[frame.handoff.isna()].set_index('method')
    selected=['cfg_tuned','cfg_high','fsg_high','fsg_length_high','fsg_debiased_high','cycle_high',
        'refined_high','smc_high','instant_high','instant01_high','soft_high','norm_high']
    selected=[key for key in selected if key in full.index]
    fig,ax=plt.subplots(figsize=(11,5))
    rows=full.loc[selected]
    ax.barh(np.arange(len(rows)),rows.fid,color=[COLORS.get(m,'#82909F') for m in selected])
    ax.set_yticks(np.arange(len(rows)),[NAMES[m] for m in selected]);ax.invert_yaxis()
    for j,value in enumerate(rows.fid):ax.text(value+.7,j,f'{value:.2f}',va='center',fontsize=9)
    ax.axvline(full.loc['cfg_tuned','fid'],color='#277DA1',ls='--',lw=1)
    ax.set(xlabel='FID (lower is better)',title='Paired 1K outputs; fixed configurations')
    ax.grid(True,axis='x');savefig(fig,'fid_mechanism_controls')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in MAIN:
        rows=frame[(frame.method==method)&(frame['tail']=='null')&frame.handoff.notna()].sort_values('handoff')
        axes[0].plot(rows.handoff/64,rows.fid,marker='o',label=NAMES[method],color=COLORS[method])
        if 'convnext_top1' in rows:
            axes[1].plot(rows.handoff/64,100*rows.convnext_top1,marker='o',label=NAMES[method],color=COLORS[method])
    axes[0].set(xlabel='NULL handoff time',ylabel='1K FID',title='Distribution quality')
    axes[1].set(xlabel='NULL handoff time',ylabel='Target top-1 (%)',title='ConvNeXt alignment',ylim=(0,100))
    for ax in axes:ax.set_xticks([.25,.5,.75]);ax.grid(True)
    axes[-1].legend(frameon=False,fontsize=8);savefig(fig,'fid_and_alignment_handoff')


def control_cycle_analysis(traces):
    """Test a closed-form prediction of the public recurrence at a frozen gap."""
    import torch
    from experiments.sit_fsg_ctrl_hypothesis_20260911 import core
    gain,decay=.2,5.
    g=torch.linspace(-.5,.5,1001,dtype=torch.float64).reshape(-1,1,1,1)
    memory=None;values=[]
    for _ in range(512):
        memory,_=core.correction(g,memory,'smc',gain,decay)
        values.append(memory.flatten().numpy().copy())
    last=np.array(values[256:]);mean=last.mean(0)
    gap=g.flatten().numpy();threshold=gain*(decay-1)/decay
    expected=np.where(np.abs(gap)<threshold,gap,gap-gain*np.sign(gap))
    eligible=(np.abs(np.abs(gap)-threshold)>1e-8)&(np.abs(gap)>1e-8)
    error=float(np.max(np.abs(mean[eligible]-expected[eligible])))
    assert error<1e-12,error
    frame=pd.DataFrame(dict(gap=gap,ctrl_mean=mean,instant=gap-gain*np.sign(gap),expected=expected,
        alternating_amplitude=(last.max(0)-last.min(0))/2,analytic_region=eligible))
    frame.to_csv(OUT/'ctrl_frozen_gap_recurrence.csv',index=False)
    atomic(OUT/'ctrl_frozen_gap_recurrence_audit.json',dict(passed=True,gain=gain,decay=decay,
        threshold=threshold,max_error_away_from_boundary=error,iterations=512,average_last=256,
        source_core_sha256=sha(core.__file__),scope='Constant-gap analytic recurrence only; actual model gaps change.'))
    settings();fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(gap/gain,gap/gain,color='#777',ls=':',label='Raw gap')
    axes[0].plot(gap/gain,frame.instant/gain,color='#287D64',label='Instantaneous')
    axes[0].plot(gap/gain,mean/gain,color='#8059A3',label='CTRL temporal mean')
    axes[0].set(xlabel='Fixed gap / K',ylabel='Modified gap / K',title='Exact recurrence at constant gap')
    axes[0].legend(frameon=False,fontsize=8);axes[0].grid(True)
    trace=traces[traces.method=='smc_high'].groupby('k')[['sign_disagreement','guidance_flip_fraction']].mean()
    axes[1].plot(trace.index[:16],100*trace.sign_disagreement.iloc[:16],marker='o',label='Correction sign differs from gap')
    axes[1].plot(trace.index[:16],100*trace.guidance_flip_fraction.iloc[:16],marker='o',label='Modified gap changes sign')
    axes[1].set(xlabel='Actual SiT step',ylabel='Coordinates (%)',title='Observed early alternation, 200 images')
    axes[1].legend(frameon=False,fontsize=8);axes[1].grid(True)
    savefig(fig,'ctrl_constant_gap_and_actual_trace')


def analyze(require_complete=False):
    OUT.mkdir(parents=True,exist_ok=True)
    p.verify()
    frame=load('trajectories');traces=load('trajectories','traces')
    interventions=load('interventions');jacobian=load('jacobian')
    if len(frame):
        assert not frame.duplicated(['method','k','suffix','index']).any()
        frame.to_csv(OUT/'trajectory_rows.csv',index=False)
        traces.to_csv(OUT/'ctrl_trace_rows.csv',index=False)
        grouped(frame,['method','k','suffix']).to_csv(OUT/'trajectory_means.csv',index=False)
        grouped(traces,['method','k']).to_csv(OUT/'ctrl_trace_means.csv',index=False)
    paired=pd.DataFrame()
    if len(interventions):
        assert not interventions.duplicated(['base','k','variant','index']).any()
        paired=paired_interventions(interventions)
        paired.to_csv(OUT/'intervention_rows.csv',index=False)
        grouped(paired,['base','k','variant']).to_csv(OUT/'intervention_means.csv',index=False)
    if len(jacobian):
        jacobian.to_csv(OUT/'jacobian_rows.csv',index=False)
        grouped(jacobian,['base','k','horizon','variant']).to_csv(OUT/'jacobian_means.csv',index=False)
    complete={part:all((p.ROOT/part/f'rank{r}.json').exists() and read(p.ROOT/part/f'rank{r}.json').get('complete') for r in range(4))
        for part in ('trajectories','interventions','jacobian')}
    if require_complete:assert all(complete.values()),complete
    if complete['trajectories']:
        assert len(frame)==23400 and len(traces)==86400
        trajectory_figures(frame,traces)
        gap_readout_correlations(frame)
        control_cycle_analysis(traces)
        paired_t=paired_trajectory(frame)
        pair_intervals(paired_t,['method','control','k','suffix'],['delta_resnet_top1','delta_convnext_top1','delta_resnet_p','delta_convnext_p']).to_csv(OUT/'trajectory_paired_intervals.csv',index=False)
        comparison_sheet(frame);comparison_sheet(frame,'null',32)
    if complete['interventions']:
        assert len(interventions)==5376,len(interventions)
        intervention_figures(paired)
        intervention_sheet(paired)
        pair_intervals(paired,['base','k','variant'],['delta_u_resnet_top1','delta_u_convnext_top1','reduction_write_rms','reduction_agreement_rms']).to_csv(OUT/'intervention_paired_intervals.csv',index=False)
    if complete['jacobian']:
        assert len(jacobian)==8832,len(jacobian)
        jacobian_figures(jacobian)
        jacobian_audit_summaries()
    if len(frame):gallery(frame,interventions)
    quality=quality_summary()
    if require_complete:
        assert len(quality)==36 and quality.complete.all(),len(quality)
        assert all(c in quality and quality[c].notna().all() for c in
            ('resnet_p','convnext_p','resnet_top1','convnext_top1')),'Quality pixel readouts are incomplete'
    if len(quality)==36 and quality.complete.all():quality_figures(quality)
    atomic(OUT/'coverage.json',dict(complete=complete,trajectory_rows=len(frame),trace_rows=len(traces),
        intervention_rows=len(interventions),jacobian_rows=len(jacobian),quality_rows=len(quality),
        quality_complete=len(quality)==36 and bool(quality.complete.all()),
        source_and_asset_hashes_verified=True))
    print(dict(complete=complete,trajectories=len(frame),interventions=len(interventions),jacobian=len(jacobian)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--score-quality',action='store_true')
    parser.add_argument('--score-quality-parallel',action='store_true')
    parser.add_argument('--rank',type=int,choices=range(4))
    args=parser.parse_args()
    if args.score_quality_parallel:score_quality_parallel()
    elif args.score_quality:score_quality(args.rank)
    else:analyze(args.require_complete)
