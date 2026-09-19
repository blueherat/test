"""Paired image and residual readouts for direct golden-path optimization."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import sit_golden_path_hypothesis_20260911 as run
from experiments.analyze_sit_fsg_ctrl_hypothesis_20260911 import font,class_names,settings,bootstrap_mean
from experiments.lifting_scale_sweep_20260909 import WORK,atomic,read,sha

OUT=WORK/'docs/data/sit_golden_path_direct_20260911'
COLORS=dict(agreement='#39846E',write='#277DA1',local='#A56C89')
NAMES=dict(agreement='Same-state future agreement',write='Frozen conditional-target transfer',local='Local velocity equality')


def load(pilot=False):
    folder=run.ROOT/('pilot' if pilot else 'runs');rows=[];traces=[];checks=[]
    run.verify()
    run.p.verify()
    request_hash=sha(run.ROOT/'request.json')
    for path in sorted(folder.glob('[0-9][0-9]_[0-9][0-9][0-9].json')):
        item=read(path)
        assert item['complete'] and item['request_sha256']==request_hash
        assert sha(path.with_suffix('.npz'))==item['state_arrays_sha256']
        rows+=item['rows'];checks+=item['gradient_checks']
        k,start=map(int,path.stem.split('_'))
        for objective,records in item['optimization_traces'].items():
            for row in records:
                for j,(mse,shift) in enumerate(zip(row['relative_mse'],row['shift_rms'])):
                    traces.append(dict(index=start+j,k=k,objective=objective,iteration=row['iteration'],
                        coarse_relative_mse=mse,shift_rms=shift,closure_calls=row['closure_calls'],
                        per_sample_rejections=row['per_sample_rejections']))
    return pd.DataFrame(rows),pd.DataFrame(traces),checks


def means(frame,keys):
    columns=[x for x in frame.select_dtypes(include=np.number).columns if x not in keys+['index','start']]
    result=frame.groupby(keys,dropna=False)[columns].mean().reset_index()
    result['n']=frame.groupby(keys,dropna=False).size().to_numpy()
    return result


def enrich(frame,traces):
    """Normalize drift and step size against the same untouched source state."""
    baseline=frame[(frame.objective=='agreement')&(frame.variant=='optimized')&(frame.iteration==0)]
    assert not baseline.duplicated(['index','k']).any()
    reference=baseline.set_index(['index','k'])[['agreement_rms','local_gap_rms','state_rms']]
    reference=reference.rename(columns={c:'initial_'+c for c in reference.columns})
    frame=frame.join(reference,on=['index','k'],validate='many_to_one')
    frame['conditional_drift_over_initial_gap']=frame.conditional_drift_rms/frame.initial_agreement_rms
    frame['guided_drift_over_initial_gap']=frame.guided_drift_rms/frame.initial_agreement_rms
    frame['shift_over_initial_state_rms']=frame.shift_rms/frame.initial_state_rms
    frame['reference_radius_rms']=(4/64)*1.25*frame.initial_local_gap_rms
    frame['shift_over_reference_radius']=frame.shift_rms/frame.reference_radius_rms
    coarse=traces.set_index(['index','k','objective','iteration'])[['coarse_relative_mse']]
    frame=frame.join(coarse,on=['index','k','objective','iteration'],validate='many_to_one')
    frame.loc[frame.iteration==0,'coarse_relative_mse']=1.
    frame.loc[frame.variant!='optimized','coarse_relative_mse']=np.nan
    frame['coarse_objective_rms_ratio']=np.sqrt(frame.coarse_relative_mse)
    frame['fine_objective_rms_ratio']=np.select([frame.objective==v for v in run.OBJECTIVES],
        [frame.agreement_ratio,frame.write_ratio,frame.local_gap_ratio],default=np.nan)
    return frame


def data_audit(frame,traces,require_complete):
    assert np.isfinite(frame.select_dtypes(include=np.number).drop(columns=[
        'coarse_relative_mse','coarse_objective_rms_ratio','fine_objective_rms_ratio'])).all().all()
    assert not traces.duplicated(['index','k','objective','iteration']).any()
    assert frame.groupby(['index','k']).size().eq(28).all()
    assert traces.groupby(['index','k','objective']).size().eq(64).all()
    max_step_increase=0.;max_shift_error=0.;files=0
    labels=np.load(run.p.ROOT/'inputs/labels.npy')
    for _,group in traces.groupby(['index','k','objective']):
        values=np.r_[1.,group.sort_values('iteration').coarse_relative_mse.to_numpy()]
        max_step_increase=max(max_step_increase,float(np.diff(values).max()))
    assert max_step_increase<=1.01e-6,max_step_increase
    # Audit precisely the snapshot loaded above while later batches may finish.
    paths=[run.ROOT/'runs'/f'{int(k):02d}_{int(start):03d}.npz'
        for k,start in frame[['k','start']].drop_duplicates().sort_values(['k','start']).itertuples(index=False,name=None)]
    for path in paths:
        k,start=map(int,path.stem.split('_'));files+=1
        with np.load(path) as data:
            np.testing.assert_array_equal(data['labels'],labels[start:start+2])
            x=data['x'].astype(np.float64)
            source_start=(start//8)*8
            source=run.study.state_path('cfg_tuned',source_start)
            assert sha(source)==read(source.with_suffix('.json'))['state_sha256']
            with np.load(source) as original:
                np.testing.assert_array_equal(x,original[f'x{k:02d}'][start-source_start:start-source_start+2])
            for objective in run.OBJECTIVES:
                for iteration in run.ITERATIONS:
                    z=data[f'{objective}_{iteration:02d}'].astype(np.float64)
                    expected=np.sqrt(np.square(z-x).reshape(2,-1).mean(1))
                    rows=frame[(frame.k==k)&(frame['index'].isin([start,start+1]))&
                        (frame.objective==objective)&(frame.iteration==iteration)&(frame.variant=='optimized')].sort_values('index')
                    assert len(rows)==2
                    max_shift_error=max(max_shift_error,float(np.max(np.abs(expected-rows.shift_rms))))
                    if iteration==0:np.testing.assert_array_equal(z,x)
    assert max_shift_error<1e-6,max_shift_error
    for keys,group in frame.groupby(['index','k','objective']):
        if keys[-1]=='frozen_inverse':continue
        full=group[(group.variant=='optimized')&(group.iteration==64)].iloc[0]
        equal=group[group.variant=='cfg_equal'].iloc[0]
        assert abs(full.shift_rms-equal.shift_rms)<1e-6
        for variant,multiple in [('radius_projected',1),('radius4_projected',4)]:
            row=group[group.variant==variant].iloc[0]
            assert row.shift_over_reference_radius<=multiple+1e-4
    for (_,k),group in frame[(frame.variant=='optimized')&(frame.iteration==0)].groupby(['index','k']):
        keys=['agreement_rms','local_gap_rms','state_rms']+[p+'_'+m for p in ('u','c','g')
            for m in ('resnet_p','convnext_p','resnet_top1','convnext_top1')]
        assert (group[keys].max()-group[keys].min()).max()<1e-6
    if require_complete:
        assert files==24 and set(frame['index'])==set(range(16))
        assert set(frame.k)==set(run.TIMES)
    atomic(OUT/'data_audit.json',dict(passed=True,complete=require_complete,
        completed_state_batches=files,source_and_array_hashes_verified=True,
        max_coarse_mse_step_increase=max_step_increase,max_saved_state_rms_error=max_shift_error,
        initial_states_identical_across_objectives=True,equal_length_controls_verified=True,
        captured_source_states_and_labels_verified=True,
        radius_projection_bounds_verified=True,image_scores_recomputed=False))


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=160)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def figures(frame):
    settings();core=frame[frame.variant=='optimized']
    grouped=means(core,['objective','iteration','k'])
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,k in zip(axes,run.TIMES):
        for objective in run.OBJECTIVES:
            part=grouped[(grouped.objective==objective)&(grouped.k==k)].sort_values('iteration')
            if not len(part):continue
            ax.plot(part.iteration,part.agreement_ratio,marker='o',color=COLORS[objective],label=NAMES[objective])
        ax.set(title=f't={k/64:g}',xlabel='Optimization iterations',ylabel='Fine C / U gap relative to initial')
        ax.set_yscale('log');ax.grid(True)
    handles,labels=next(((ax.get_legend_handles_labels()) for ax in axes if ax.lines),([],[]))
    if handles:axes[-1].legend(handles,labels,frameon=False,fontsize=7)
    fig.suptitle('Full latent optimization; all outcomes re-evaluated on the original fine grid')
    fig.tight_layout();save(fig,'fine_future_agreement')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    aggregate=means(core,['objective','iteration'])
    for ax,prefix,title in zip(axes,['u','c','g'],['NULL completion','Conditional completion','CFG completion']):
        for objective in run.OBJECTIVES:
            part=aggregate[aggregate.objective==objective].sort_values('iteration')
            if not len(part):continue
            ax.plot(part.iteration,part[prefix+'_convnext_p'],marker='o',color=COLORS[objective],label=NAMES[objective])
        ax.set(title=title,xlabel='Optimization iterations',ylabel='ConvNeXt target probability',ylim=(0,1));ax.grid(True)
    axes[-1].legend(frameon=False,fontsize=7)
    fig.suptitle('Posthoc readouts; classifiers do not participate in optimization')
    fig.tight_layout();save(fig,'target_readouts_by_objective')
    fig,axes=plt.subplots(3,3,figsize=(13.5,10),sharex=True,sharey=True)
    for j,k in enumerate(run.TIMES):
        for i,(prefix,title) in enumerate(zip(['u','c','g'],['NULL','Conditional','CFG'])):
            ax=axes[i,j]
            for objective in run.OBJECTIVES:
                part=grouped[(grouped.objective==objective)&(grouped.k==k)].sort_values('iteration')
                if not len(part):continue
                ax.plot(part.iteration,part[prefix+'_convnext_p'],marker='o',color=COLORS[objective],label=NAMES[objective])
            ax.set(title=f'{title} completion; t={k/64:g}',ylim=(0,1));ax.grid(True)
            if i==2:ax.set_xlabel('Optimization iterations')
            if j==0:ax.set_ylabel('Target probability (ConvNeXt)')
    handles,labels=next(((ax.get_legend_handles_labels()) for ax in axes.flat if ax.lines),([],[]))
    if handles:fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Same fixed samples at each time; posthoc category readouts')
    fig.tight_layout(rect=[0,.04,1,.97]);save(fig,'target_readouts_by_time')


def sheet(frame,objective='agreement',k=24,iteration=64):
    indices=(0,1,2,3)
    subset=frame[(frame.objective==objective)&(frame.k==k)&(frame.variant=='optimized')]
    if not set(indices).issubset(set(subset[subset.iteration==iteration]['index'])):return None
    labels=np.load(run.p.ROOT/'inputs/labels.npy');names=class_names()
    columns=[(0,'u','Original NULL'),(0,'c','Original conditional'),(0,'g','Original CFG'),
        (iteration,'u','After: NULL'),(iteration,'c','After: conditional'),(iteration,'g','After: CFG')]
    tile,pad,left,top,height=256,12,220,94,333
    image=Image.new('RGB',(left+len(columns)*(tile+pad)+pad,top+4*height+pad),'white')
    draw=ImageDraw.Draw(image)
    draw.text((12,8),f'{NAMES[objective]} · t={k/64:g} · {iteration} iterations · first four fixed samples',font=font(23),fill='#17212b')
    for j,(_,_,title) in enumerate(columns):draw.text((left+j*(tile+pad),53),title,font=font(18),fill='#17212b')
    for i,index in enumerate(indices):
        y0=top+i*height;draw.text((12,y0+18),f'#{index:03d}',font=font(23),fill='#17212b')
        words=names[int(labels[index])].split();lines=['']
        for word in words:
            if len(lines[-1]+' '+word)>17:lines.append(word)
            else:lines[-1]=(lines[-1]+' '+word).strip()
        for n,line in enumerate(lines):draw.text((12,y0+55+24*n),line,font=font(18),fill='#17212b')
        for j,(step,prefix,_) in enumerate(columns):
            row=subset[(subset['index']==index)&(subset.iteration==step)].iloc[0]
            path=run.ROOT/'images'/f'{k:02d}'/f'{objective}_{step:02d}'/prefix/f'{index:03d}.png'
            pixels=Image.open(path).convert('RGB');assert pixels.size==(256,256)
            x0=left+j*(tile+pad);image.paste(pixels,(x0,y0))
            draw.text((x0,y0+261),f'CN {row[prefix+"_convnext_p"]:.3f} · R18 {row[prefix+"_resnet_p"]:.3f}',font=font(14),fill='#334155')
            draw.text((x0,y0+283),f'C/U gap ratio {row.agreement_ratio:.3f}',font=font(14),fill='#334155')
            draw.text((x0,y0+305),f'State shift RMS {row.shift_rms:.3f}',font=font(14),fill='#334155')
    path=OUT/f'examples_{objective}_{k:02d}_{iteration:02d}.png';image.save(path);return path


def gallery(frame):
    labels=np.load(run.p.ROOT/'inputs/labels.npy');names=class_names()
    data=json.dumps(dict(rows=frame.to_dict('records'),names=NAMES,
        samples=[dict(index=i,label=names[int(labels[i])]) for i in range(16)]),ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>直接检验 U(x)=C(x)</title>
<style>body{font:15px system-ui,sans-serif;margin:24px;background:#f6f8fa;color:#17212b}h1{font-size:25px}p{max-width:1150px;line-height:1.7}nav{display:flex;gap:20px;flex-wrap:wrap}select{display:block;padding:9px;margin-top:5px;min-width:170px}table{border-collapse:separate;border-spacing:10px}th{vertical-align:top;text-align:left;min-width:160px}td{padding:10px;background:white;border:1px solid #d6dee7;border-radius:5px}img{display:block;width:256px;height:256px}small{line-height:1.6;color:#526270}.scroll{overflow:auto}</style>
<h1>同一状态的条件 / NULL 未来一致性：直接优化实验</h1><p>固定前16个噪声；完整4096维latent优化，模型权重冻结。优化12步未来，图中重新运行原64网格的剩余段。分类器只读取最终像素。这里没有16张FID，也没有把残差下降自动解释成质量改善。</p>
<p>“未来一致性”同时允许条件与NULL结局变化；“冻结条件目标”固定修改前的条件结局；“局部速度相等”只优化当前两个预测。大位移无法保证仍处于合适的概率区域，半径投影对照另列。点击图片查看原始像素。</p>
<nav><label>样本 / 目标<select id="sample"></select></label><label>时刻<select id="time"><option value="24">t=0.375</option><option value="8">t=0.125</option><option value="40">t=0.625</option></select></label><label>优化目标<select id="objective"><option value="agreement">同一状态的未来一致性</option><option value="write">冻结条件目标的转移</option><option value="local">局部速度相等</option></select></label></nav><div class="scroll" id="view"></div>
<script>const D=__DATA__,s=document.querySelector('#sample'),t=document.querySelector('#time'),o=document.querySelector('#objective');D.samples.forEach(r=>{let e=document.createElement('option');e.value=r.index;e.textContent=`#${String(r.index).padStart(3,'0')} · ${r.label}`;s.append(e)});
function draw(){const i=+s.value,k=+t.value,obj=o.value;let rows=D.rows.filter(r=>r.index===i&&r.k===k&&(r.objective===obj||r.objective==='frozen_inverse')),body='<tr><th>更新 / 细网格结果</th><th>NULL结局</th><th>条件结局</th><th>CFG结局</th></tr>';
for(const r of rows){let name=r.variant==='optimized'?`迭代 ${r.iteration}`:r.variant,dir=r.objective==='frozen_inverse'?'frozen_inverse':(r.variant==='optimized'?`${r.objective}_${String(r.iteration).padStart(2,'0')}`:`${r.objective}_${r.variant}`);body+=`<tr><th>${name}<br><small>未来差/初值 ${r.agreement_ratio.toFixed(3)}<br>写入误差/初值 ${r.write_ratio.toFixed(3)}<br>局部gap/初值 ${r.local_gap_ratio.toFixed(3)}<br>条件终点漂移/原差 ${r.conditional_drift_over_initial_gap.toFixed(3)}<br>位移RMS ${r.shift_rms.toFixed(3)}<br>位移/参考半径 ${r.shift_over_reference_radius.toFixed(1)}×</small></th>`;for(const prefix of ['u','c','g']){let path=`images/${String(k).padStart(2,'0')}/${dir}/${prefix}/${String(i).padStart(3,'0')}.png`;body+=`<td><a href="${path}" target="_blank"><img loading="lazy" src="${path}" alt="同状态干预的真实生成图片"></a><small>R18 ${r[prefix+'_resnet_p'].toFixed(3)} · CN ${r[prefix+'_convnext_p'].toFixed(3)}</small></td>`}body+='</tr>'}document.querySelector('#view').innerHTML=rows.length?'<table>'+body+'</table>':'当前组合尚未完成';}for(const e of [s,t,o])e.addEventListener('change',draw);draw();</script></html>'''
    (run.ROOT/'gallery.html').write_text(html.replace('__DATA__',data))


def analyze(require_complete=False):
    OUT.mkdir(parents=True,exist_ok=True);frame,traces,checks=load()
    if not len(frame):print('No complete batches yet',flush=True);return
    assert not frame.duplicated(['index','k','objective','iteration','variant']).any()
    if require_complete:
        assert (run.ROOT/'complete.json').exists()
        assert len(frame)==1344 and len(traces)==9216,(len(frame),len(traces))
    frame=enrich(frame,traces)
    data_audit(frame,traces,require_complete)
    baseline=frame[(frame.variant=='optimized')&(frame.iteration==0)].set_index(['index','k','objective'])
    paired=frame[frame.objective!='frozen_inverse'].join(baseline[[p+'_'+c for p in ('u','c','g')
        for c in ('resnet_p','resnet_top1','convnext_p','convnext_top1')]],on=['index','k','objective'],rsuffix='_before',validate='many_to_one')
    for prefix in ('u','c','g'):
        for key in ('resnet_p','resnet_top1','convnext_p','convnext_top1'):
            metric=prefix+'_'+key;paired['delta_'+metric]=paired[metric]-paired[metric+'_before']
    frame.to_csv(OUT/'rows.csv',index=False);traces.to_csv(OUT/'optimization_traces.csv',index=False)
    paired.to_csv(OUT/'paired_rows.csv',index=False)
    means(frame,['k','objective','iteration','variant']).to_csv(OUT/'means.csv',index=False)
    means(paired,['objective','iteration','variant']).to_csv(OUT/'aggregate_means.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'gradient_checks.csv',index=False)
    figures(frame)
    for objective in run.OBJECTIVES:
        for k in run.TIMES:sheet(frame,objective,k)
    gallery(frame)
    atomic(OUT/'coverage.json',dict(complete=bool((run.ROOT/'complete.json').exists()),
        rows=len(frame),trace_rows=len(traces),unique_noises=int(frame['index'].nunique()),
        fine_equality_ratio_mean=float(frame[(frame.objective=='agreement')&(frame.iteration==64)&(frame.variant=='optimized')].agreement_ratio.mean())
        if ((frame.objective=='agreement')&(frame.iteration==64)&(frame.variant=='optimized')).any() else None))
    print(dict(rows=len(frame),unique_noises=frame['index'].nunique(),checks=len(checks)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true')
    analyze(parser.parse_args().require_complete)
