"""Complete fixed reflection experiment with all controls and primary-source limits."""
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import openpyxl
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import audit,mixture
from . import core as m

OUT=c.WORK/'docs/data/reflection_guidance_20260912'
REPORT=c.WORK/'docs/REFLECTION_GUIDANCE_RESULTS_20260912_ZH.md'
NAMES={'sit_small':'SiT-S/2','raev2':'RAEv2'}


def finalize():
    m.configure();mixture.ROOT=m.ROOT;torch.set_num_threads(4)
    status=m.ROOT/'status.json'
    while c.read(status)['phase']!='complete':
        r=c.read(status)
        if not Path('/proc',str(r['pid'])).exists():raise RuntimeError('Controller terminal before completion')
        time.sleep(5)
    for model in c.MODELS:
        while not (m.ROOT/model/'inference_benchmark.json').exists():time.sleep(5)
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];decisions=[];audits=[];bench=[]
    for model in c.MODELS:
        m.verify(model)
        results=c.read(m.ROOT/model/m.STAGE/'results.json');assert len(results)==len(m.ARMS)
        by={r['arm']:r for r in results}
        for r in results:
            rows.append({k:r[k] for k in ('model','arm','fid','inception_score','seconds','primary_samples',
                'generated_paths','full_calls_per_output','prefix_calls_at_inference')})
            audits.append(audit.arm(model,m.STAGE,r['arm']))
        for track in ('cfg','ig'):
            candidate=by[track+'_abba'];controls=[by[track+'_'+s] for s in ('native','half','fixed_flip')]
            best=min(controls,key=lambda r:r['fid']);native=by[track+'_native']
            decisions.append(dict(model=model,track=track,candidate_fid=candidate['fid'],
                best_control=best['arm'],best_control_fid=best['fid'],delta=candidate['fid']-best['fid'],
                inception_ratio=candidate['inception_score']/native['inception_score'],
                passes=candidate['fid']<=best['fid']-2 and candidate['inception_score']>=.9*native['inception_score'],
                strong_abba_minus_native=by['strong_abba']['fid']-by['strong_native']['fid']))
        for r in c.read(m.ROOT/model/'inference_benchmark.json')['records']:
            bench.append(dict(model=model,track=r['track'],batch=r['batch'],native_seconds=r['median'][r['track']+'_native'],
                candidate_seconds=r['median'][r['track']+'_abba'],relative_change=r['relative_median_change']))
    tables=dict(quality=pd.DataFrame(rows),decisions=pd.DataFrame(decisions),timing=pd.DataFrame(bench))
    sources=pd.DataFrame([
        dict(title='Frozen reflection protocol',url=str(m.PROTOCOL),scope='Arms, bank and promotion rule'),
        dict(title='Raw generation and input hash records',url=str(m.ROOT),scope='22 arms, 8800 single-path images'),
        dict(title='Visual Anagrams (Geng et al., CVPR 2024)',url='https://arxiv.org/html/2311.17919v2',scope='View fusion, alternating baseline, latent caveat'),
        dict(title='SymDiff (Zhang et al., ICLR 2025)',url='https://arxiv.org/abs/2410.06262',scope='Prior stochastic symmetrisation at sampling time'),
        dict(title='Rao-Blackwell Gradient Estimators for Equivariant Denoising Diffusion (Tong et al., v2 2025)',
            url='https://arxiv.org/html/2502.09890v2',scope='Prior training target symmetrisation; different from this frozen sampler')])
    for name,df in tables.items():df.to_csv(OUT/(name+'.csv'),index=False)
    with pd.ExcelWriter(OUT/'source_data.xlsx',engine='openpyxl') as w:
        for name,df in dict(tables,Sources=sources).items():df.to_excel(w,sheet_name=name,index=False)
        for sh in w.book.worksheets:
            sh.freeze_panes='A2';sh.auto_filter.ref=sh.dimensions
            for col in sh.columns:sh.column_dimensions[col[0].column_letter].width=min(65,max(16,max(len(str(v.value or '')) for v in col)+2))
    q=tables['quality'];fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for i,model in enumerate(c.MODELS):
        for j,track in enumerate(('cfg','ig')):
            df=q[(q.model==model)&q.arm.str.startswith(track)].set_index('arm')
            order=[track+'_'+s for s in ('native','half','fixed_flip','abba')]+([track+'_mismatched'] if track=='cfg' else [])
            df=df.loc[order];ax=axes[i,j];delta=df.fid-df.loc[track+'_native'].fid
            ax.barh(np.arange(len(df)),delta,color=['#888888' if not s.endswith('abba') else '#176b8d' for s in order])
            ax.set_yticks(np.arange(len(df)),[s[len(track)+1:] for s in order]);ax.invert_yaxis()
            ax.axvline(0,color='black',lw=.8);ax.set_xlabel('FID difference from native; lower is better')
            ax.set_title(NAMES[model]+' '+track.upper()+' · 400 images')
            lo=min(-2,float(delta.min())-3);hi=max(8,float(delta.max())+12);ax.set_xlim(lo,hi)
            for k,r in enumerate(df.itertuples()):ax.text(hi-.02*(hi-lo),k,f'FID {r.fid:.2f}',ha='right',va='center',fontsize=9)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/('quality_comparison.'+ext),dpi=180)
    plt.close(fig)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for model in c.MODELS:
        for track in ('cfg','ig'):
            kinds=[track+'_native',track+'_fixed_flip',track+'_abba']
            width,height=180+192*4,45+210*len(kinds)
            canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
            draw.text((10,8),NAMES[model]+' '+track.upper()+': fixed first four; no selection',font=font,fill='black')
            for i,kind in enumerate(kinds):
                root=m.ROOT/model/m.STAGE/kind
                with np.load(root/'samples.npz') as d:pixels=d['arr_0'][:4]
                draw.text((10,70+i*210),kind,font=font,fill='black')
                draw.text((10,93+i*210),f'FID {c.read(root/"metrics.json")["fid"]:.2f}',font=font,fill='black')
                for j,img in enumerate(pixels):canvas.paste(Image.fromarray(img).resize((192,192)),(180+192*j,45+i*210))
            canvas.save(OUT/(model+'_'+track+'_first4.png'))
    decision=tables['decisions'];eligible=decision[decision.passes]
    txt=['# 加性预测错配与固定预算反射交替\n\n',
        '本轮检验整个引导场在原视图和水平反射视图之间固定ABBA交替，是否能在原CFG／IG调用预算内抑制一部分加性预测误差。'
        '两个模型各11臂，每臂400个单路径图像，共8800图；CFG和IG独立判断，使用同一全新输入bank。'
        '未训练新模型、未增加弱前缀，也未搜索周期、相位、强度或截止时间。\n\n',
        f'按预先冻结的筛选规则，进入全新1K确认的组数为 **{len(eligible)}/4**。400样本特别是RAE的400类别子集不足以支持最终统计结论。\n\n',
        decision.to_markdown(index=False)+'\n\n',
        '四组ABBA的FID都高于各自最佳固定对照，SiT CFG／IG分别退化约22.75／30.81，RAEv2分别约2.99／3.51。'
        'SiT的IS也明显降低。无引导Strong ABBA同样退化，说明本次变化没有获得可用于guidance的通用采样收益。'
        '本轮结束当前固定latent反射构造，未启动1K／5K晋级，不搜索其他周期、相位或强度。\n\n',
        '![全部CFG和IG比较](data/reflection_guidance_20260912/quality_comparison.png)\n\n',
        '## 解释性假设及其范围\n\n',
        '设反射作用为R，$F^R(z)=R^{-1}F(Rz)$。若理想场满足该等变关系，且误差评价的状态测度也反射不变，'
        '群平均$(F+F^R)/2$是L2意义下的正交投影，不能增加场误差。这只去除不等变部分；共同污染或加性误差中的等变部分仍会保留。'
        '它没有要求强弱全部误差共线，也没有识别一般未知残差。\n\n',
        '实际采样没有在同一个状态额外计算两个视图，而是每四个完整步用A、B、B、A。'
        '这只是在有限步下近似平均场。SiT同一Heun步保持同一视图，RAE保留非均匀Euler网格；'
        '后者不因使用回文顺序就获得二阶精度。反射步的总时间质量SiT为0.5、RAE约0.49727。\n\n',
        '理论要求的latent分布对称性也没有保证：SD-VAE编码器和DINO表示不必与像素水平反射交换。'
        '即使单视图反射能解码出图像，交替后的场也可能改变模式覆盖，或引入新的表示偏差。'
        'L2场误差结论不能替代生成FID、条件正确性或多样性检验。\n\n',
        '## 固定反射、错配视图和Strong对照\n\n',
        '固定反射保持同一变换直到终点，已逐位验证等于对反射初始噪声运行原采样再反射latent输出。'
        '它有助于区分单次坐标变换与反复切换的影响。图像最终仍由原解码器得到，没有假设解码也严格交换。\n\n',
        '固定反射退化后追加的[同端点解码检查](REFLECTION_REPRESENTATION_REVIEW_20260912_ZH.md)已经完成：'
        '先还原latent、解码后再反射像素，SiT CFG／IG的FID从140.61／164.75变为77.68／103.98，'
        'RAEv2从105.31／114.89变为97.71／96.04。新增1600个渲染视图，采样full／prefix调用均为0。'
        '它支持表示与解码不交换对固定反射的影响，不能归因ABBA的全部退化，也不能作为新方法正结果；原候选筛选未改动。\n\n',
        'CFG错配对照让conditional和null使用相反视图。若二者共同的不等变误差分别为±e，'
        '匹配视图的组合只留下±e，错配视图的组合会留下±(1+2g)e。'
        '不过匹配与错配两种方案具有相同的理想双视图平均场；差异只能通过有限步的切换、状态传播及非理想误差体现。'
        '因此不能用它们的FID差证明不同的连续终点密度目标。\n\n',
        '无引导ABBA也改变Strong的原采样，所以零guidance并不恢复原Strong轨迹。本轮显式保留Strong两臂，'
        '不把通用采样变化自动归因于guidance机制。两种Strong的FID与IS如下。\n\n',
        q[q.arm.str.startswith('strong')][['model','arm','fid','inception_score']].to_markdown(index=False)+'\n\n',
        '## 实际生成结果与成本\n\n',q.to_markdown(index=False)+'\n\n',
        'CFG保持SiT224、RAE200次full，IG保持128、100次full；额外prefix均为0。'
        '每个单路径输出独立进入主评价，不使用双分支重复样本。质量采样总秒数是各batch采样和解码耗时之和，'
        '不是多GPU墙钟时间；模型加载、预检及CPU特征评价另属执行成本。\n\n',
        tables['timing'].to_markdown(index=False)+'\n\n',
        '计时使用同卡、同批量、3次原方法／ABBA交替重复，包含解码。它只提供小规模开销估计。'
        '原方法包装与直接实现、固定反射的完整轨迹关系均逐位通过；没有通过增加隐藏前向取得效果。\n\n',
        '## 与已有工作的关系\n\n',
        '[Visual Anagrams](https://arxiv.org/html/2311.17919v2)已经研究变换后的预测平均及按步交替，在其错觉生成设置中平均优于交替，并讨论latent变换伪影。'
        '本轮保持同一条件，以生成质量和原调用预算为目标，但不据此认领“交替视图”本身。\n\n',
        '[SymDiff](https://arxiv.org/abs/2410.06262)已经将随机对称化用于扩散生成。'
        '[Tong等的v2论文](https://arxiv.org/html/2502.09890v2)研究群轨道条件期望降低训练梯度方差；'
        '其v2标题为Rao-Blackwell Gradient Estimators for Equivariant Denoising Diffusion，不能只引用旧标题而忽略版本变化。'
        '本轮不把冻结模型上的一个采样实例等同于这些训练方法的新发现。\n\n',
        '仓库9月10日第24项只对IG差值做额外反射平均，未超过原IG。此次针对整个场、共享既有时间查询的检验改变了预算和作用对象，'
        '但仍需要实际收益才能继续。生成未通过则停止当前反射构造，不用群平均的理论继续为它调参数。\n\n',
        '## 固定样本与可复核数据\n\n']
    for model in c.MODELS:
        for track in ('cfg','ig'):txt.append(f'![{NAMES[model]} {track.upper()}](data/reflection_guidance_20260912/{model}_{track}_first4.png)\n\n')
    txt += ['[冻结协议](REFLECTION_GUIDANCE_PROTOCOL_20260912_ZH.md) · [完整工作簿](data/reflection_guidance_20260912/source_data.xlsx) · '
        '[全部指标](data/reflection_guidance_20260912/quality.csv) · [最终核验](data/reflection_guidance_20260912/verification.json)。\n\n',
        '共同污染＋结构化错配＋加性残差的主假设继续保留；本轮只检验了一个反射处理实例，'
        '没有验证一般残差可识别，也没有据此完成具有可靠同预算收益与实质新意的长期目标。\n']
    REPORT.write_text(''.join(txt))
    w=openpyxl.load_workbook(OUT/'source_data.xlsx',read_only=True);assert w['quality'].max_row==23;w.close()
    c.atomic(OUT/'verification.json',dict(passed=True,arms=22,generated_images=8800,
        max_fid_recalculation_error=max(r['absolute_error'] for r in audits),same_feature_recalculation_only=True,
        audits=audits,eligible_for_1k=eligible[['model','track']].to_dict(orient='records'),
        visual_inspection_pending=True,goal_complete=False))
    c.atomic(OUT/'artifact_manifest.json',dict(files={str(p.relative_to(c.WORK)):c.sha(p) for p in OUT.iterdir() if p.name!='artifact_manifest.json'},
        report_sha256=c.sha(REPORT),generator_sha256=c.sha(Path(__file__))))
    print('Finalized',REPORT,'eligible',eligible[['model','track']].to_dict(orient='records'),flush=True)


if __name__=='__main__':finalize()
