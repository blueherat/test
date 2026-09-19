"""Audit every SG search point and all fifteen independent 5K arms."""
import csv,shutil
from pathlib import Path
import numpy as np
import torch
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from experiments.ig_sg_5k_20260914 import grid as r
from experiments.guidance_pasted_20260912.audit import fid_from_features
from experiments.raev2_context_5k_20260914.report import fid_from_covariance
c=r.c
OUT=c.WORK/'docs/data/ig_sg_5k_20260914'
REPORT=c.WORK/'docs/IG_SG_5K_RESULTS_20260914_ZH.md'
DISPLAY={'sit_small':'SiT-S/2','jit':'JiT-B/16','raev2':'RAEv2'}


def reference(path):
    with np.load(path) as d:
        mu,cov=(d['mu'],d['sigma']) if 'mu' in d else (d['ref_mu'],d['ref_sigma'])
    mu,cov=np.asarray(mu,np.float64),np.asarray(cov,np.float64)
    eig,u=linalg.eigh((cov+cov.T)*.5,check_finite=True);assert eig.min()>-1e-8
    root=(u*np.sqrt(np.maximum(eig,0)))@u.T
    return mu,cov,root


def expected(cfg):
    stages=2 if cfg['solver']=='heun' else 1
    total=stages*cfg['steps']*{'baseline':1,'sg':2,'log':3}[cfg['kind']]
    if cfg['kind']=='sg':total-=1
    if cfg['kind']=='log' and stages==2:total-=2
    return total


def audit(phase,name,ref):
    req=r.verify(phase,name);root=r.ROOT/phase/name;rh=c.sha(root/'request.json')
    noise=np.load(req['noise'],mmap_mode='r');labels=np.load(req['labels']);n=req['samples']
    assert len(noise)==len(labels)==n
    classes=100 if name=='sit_small' else 1000
    assert np.array_equal(np.bincount(labels,minlength=classes),np.full(classes,n//classes))
    rows=[];audits=[];out=OUT/phase/name;out.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(230+6*144,35+len(req['configs'])*165),'white') if phase=='confirm_5k' else None
    if canvas:draw=ImageDraw.Draw(canvas);draw.text((8,8),name+': first six fixed paired IDs, no sample selection',fill='black')
    for i,cfg in enumerate(req['configs']):
        folder=root/cfg['arm']
        if (folder/'invalid.json').exists():
            failure=c.read(folder/'invalid.json');assert failure['invalid'] and failure['request_sha256']==rh
            shutil.copy2(folder/'invalid.json',out/f"{cfg['arm']}_invalid.json");continue
        summary=c.read(folder/'summary.json');metric=c.read(folder/'metrics.json')[0]
        assert summary['complete'] and summary['samples']==n and summary['request_sha256']==rh
        assert summary['full_calls_per_output']==expected(cfg) and summary['prefix_calls_per_output']==0
        assert c.sha(folder/'samples.npz')==summary['samples_sha256']
        if name!='sit_small':assert metric['sample_sha256']==summary['samples_sha256']
        origin=summary.get('original_source');batch_request=origin['original_request_sha256'] if origin else rh
        with np.load(folder/'samples.npz') as d:images=d['arr_0']
        assert images.shape==(n,256,256,3) and images.dtype==np.uint8
        coverage=[];seconds=0.
        for record in summary['records']:
            path=Path(record['path']);assert c.sha(path)==record['sha256']
            with np.load(path) as d:
                start=int(d['start']);stop=start+len(d['pixels']);coverage.extend(range(start,stop))
                assert str(d['request_sha256'])==batch_request
                assert str(d['noise_sha256'])==c.array_sha(noise[start:stop])
                assert np.array_equal(d['labels'],labels[start:stop]) and np.array_equal(d['pixels'],images[start:stop])
                r.g.check_latents(d,(stop-start,*r.m.SHAPES[name]))
                assert int(d['full'])==expected(cfg)
                if 'prefix' in d:assert int(d['prefix'])==0
                if 'trace' in d:assert np.isfinite(d['trace']).all()
                seconds+=float(d['seconds'])
        assert coverage==list(range(n)) and abs(seconds-summary['seconds'])<1e-5
        if name=='sit_small':
            fp=folder/'inception_activations.npz'
            with np.load(fp) as d:features=d['pool_3']
        else:
            paths=list((folder/'features').glob('*.features.pt'));assert len(paths)==1;fp=paths[0]
            features=torch.load(fp,map_location='cpu',weights_only=True).numpy()
        assert features.shape==(n,2048) and np.isfinite(features).all()
        fid=fid_from_features(features,ref[0],ref[1]) if n<2048 else fid_from_covariance(features,ref)
        error=abs(fid-metric['fid']);assert error<.002,(phase,name,cfg,error)
        audits.append(dict(arm=cfg['arm'],passed=True,coverage=n,fid_fp64=fid,error=error,
            independent_extractor=False,features_sha256=c.sha(fp),samples_sha256=summary['samples_sha256'],
            original_batch_request_sha256=batch_request))
        row=dict(cfg,phase=phase,model=name,samples=n,fid=metric['fid'],sfid=metric.get('sfid'),inception_score=metric['inception_score'],
            w=1+cfg.get('omega',0),full_calls=expected(cfg),seconds=seconds,reused=bool(origin),source_output=str(folder))
        rows.append(row);shutil.copy2(folder/'metrics.json',out/f"{cfg['arm']}_metrics.json")
        if canvas:
            top=35+i*165;draw.text((5,top+10),cfg['arm'],fill='black')
            draw.text((5,top+30),f"FID {row['fid']:.3f}; NFE {row['full_calls']}",fill='black')
            for j,pixel in enumerate(images[:6]):
                x=230+144*j;canvas.paste(Image.fromarray(pixel).resize((140,140)),(x,top));draw.text((x,top+141),f'ID {j}; class {labels[j]}',fill='black')
    if canvas:canvas.save(out/'comparison.png')
    c.atomic(out/'results.json',dict(complete=True,rows=rows,audits=audits,request_sha256=rh,
        class_counts=np.bincount(labels,minlength=100 if name=='sit_small' else 1000).tolist()))
    shutil.copy2(root/'request.json',out/'request.json');print('Audited',phase,name,len(rows),flush=True)
    return rows,audits


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    assert c.read(r.ROOT/'pipeline_complete.json')['complete']
    refs={};all_rows=[];all_audits=[]
    for name in r.MODELS:
        req=r.verify('confirm_5k',name);path=req['reference']
        if path not in refs:refs[path]=reference(path)
        for phase in r.STAGES:
            assert c.read(r.ROOT/phase/'controller_complete.json')['complete']
            rows,audits=audit(phase,name,refs[path]);all_rows+=rows;all_audits+=audits
    fields=['phase','model','arm','kind','w','omega','steps','solver','samples','fid','sfid','inception_score','full_calls','seconds','reused','source_output']
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_rows)
    for name in r.MODELS:
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        for ax,phases,n in [(axes[0],('grid_tune',),5000),(axes[1],('grid_tune','grid_refine'),5000)]:
            rows=[x for x in all_rows if x['model']==name and x['phase'] in phases]
            baseline=next(x for x in rows if x['kind']=='baseline');ax.axhline(baseline['fid'],color='black',ls='--',label='IG only')
            for kind,label in [('sg','Paper SG'),('log','Log-score SG')]:
                values=sorted([x for x in rows if x['kind']==kind],key=lambda x:x['w'])
                ax.plot([x['w'] for x in values],[x['fid'] for x in values],'-o',label=label,markersize=4)
            ax.set(xlabel='SG extrapolation w (omega = w - 1)',ylabel=f'FID, {n} paired selection images',title=DISPLAY[name])
            ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUT/f'{name}_search.png',dpi=160);fig.savefig(OUT/f'{name}_search.pdf');plt.close(fig)
    final=[x for x in all_rows if x['phase']=='confirm_5k'];assert len(final)==15 and all(x['samples']==5000 and not x['reused'] for x in final)
    lines=['# IG＋SG：先搜索外推系数，再独立5K比较','',
        '**已完成三个模型、两种SG的系数扫描及全部15组独立5K，共75,000张全新最终质量样本。** 粗扫与局部细化中的每个外推系数都直接使用5K，再用独立新种子的5K确认。1K排名不参与候选筛选。训练、头层位和IG系数沿用此前各模型的冻结配置；RAEv2使用原生第8层IG、w=1.78。','',
        '外推定义为v_out=v_IG+(w_SG−1)×(v_strong−v_reference)，代码omega=w_SG−1。扫描的是SG外推强度；此前omega1就是w_SG=2。原版SG保持朝更噪时刻偏移.01；log-score保持同时间对称扰动kappa=.2。每个方法选本次正外推搜索范围内的最低FID设置，IG-only始终单独比较；不能把最优候选仍差于IG的情况称为SG收益。','',
        '| 模型 | IG | 原版SG最佳正系数w / 5K FID↓ | 原版SG成本对照 | log-score最佳正系数w / 5K FID↓ | log-score成本对照 |',
        '|---|---:|---:|---:|---:|---:|']
    invalid=[dict(c.read(p),path=str(p)) for phase in ('grid_tune','grid_refine') for p in sorted((r.ROOT/phase).glob('*/*/invalid.json'))]
    if invalid:lines+=['','数值检查失败、未计算部分样本FID的系数：'+ '；'.join(x['path'] for x in invalid),'']
    comparisons=[]
    for name in r.MODELS:
        values=[x for x in final if x['model']==name];by={x['arm']:x for x in values}
        base=by['ig'];sg=next(x for x in values if x['kind']=='sg');log=next(x for x in values if x['kind']=='log')
        sc,lc=by['ig_sg_cost'],by['ig_log_cost']
        lines.append(f"| {DISPLAY[name]} | {base['fid']:.4f} | {sg['w']:.4f} / {sg['fid']:.4f} | {sc['fid']:.4f} | {log['w']:.4f} / {log['fid']:.4f} | {lc['fid']:.4f} |")
        comparisons.append(dict(model=name,sg_w=sg['w'],log_w=log['w'],sg_delta_ig=sg['fid']-base['fid'],sg_delta_cost=sg['fid']-sc['fid'],
            log_delta_ig=log['fid']-base['fid'],log_delta_cost=log['fid']-lc['fid'],sg_seconds_ratio=sg['seconds']/base['seconds'],log_seconds_ratio=log['seconds']/base['seconds']))
    lines+=['','差值为候选FID减对照，负数才是改善：','',
        '| 模型 | 原版SG对IG | 原版SG对同成本 | log-score对IG | log-score对同成本 | 原版SG耗时比 | log-score耗时比 |','|---|---:|---:|---:|---:|---:|---:|']
    for x in comparisons:lines.append(f"| {DISPLAY[x['model']]} | {x['sg_delta_ig']:+.4f} | {x['sg_delta_cost']:+.4f} | {x['log_delta_ig']:+.4f} | {x['log_delta_cost']:+.4f} | {x['sg_seconds_ratio']:.2f}× | {x['log_seconds_ratio']:.2f}× |")
    lines+=['','SiT Heun64：IG128次、原版SG255次、log-score382次；对照Heun128为256次、Heun191为382次。JiT／RAEv2 Euler100：100／199／300次，对照Euler199／300。SiT原版SG成本对照多一次前向，其他完全一致。耗时为共享机器队列中的采样和解码墙钟时间，排除载入、预热、写盘与评价；它不是独占机器速度基准，计算量以实际NFE为准。','',
        '每个模型先在w_SG=1至2按0.2扫描，再对每种SG按两端5K FID均值选择前三个相邻区间，以0.05补齐内部点。每个粗扫与细化点均直接生成5000张，使用seed2026091441完整类别均衡bank。此前1K仅作历史诊断，不据其排名淘汰候选，也不混入当前调参指标。','',
        '最终5K为全新seed2026091452；SiT的100类各50张，JiT／RAEv2的1000类各5张，同模型全部配置共享噪声和标签。SiT使用ADM ImageNet100参考，其他模型用Nanogen ImageNet256参考；两类协议的绝对FID不跨模型比较。','',
        '“最佳”只表示已测范围内在调参集上的最低FID，5K负责检验这一选择。这里只有一个最终5K bank，微小差异仍不自动构成显著改善。原版SG按发布代码做velocity外推；log-score平滑的是score／log-density，并非密度高斯卷积后的精确score。','',
        '[RAEv2后训练头的独立报告](RAEV2_SHALLOW_IG_5K_SELECTION_RESULTS_20260914_ZH.md)使用不同5K噪声bank，不能拿两份表里的细微RAEv2差异直接排序。','']
    for name in r.MODELS:
        lines += [f'## {DISPLAY[name]} 的完整扫描与固定配对图','',f'![外推系数扫描](data/ig_sg_5k_20260914/{name}_search.png)','',
            f'![固定前6个ID](data/ig_sg_5k_20260914/confirm_5k/{name}/comparison.png)','']
    lines+=['## 复核与来源','',
        '全部扫描点和最终样本保留。逐批验证输入、标签、样本覆盖、全部合并像素、原始request身份与实际NFE；调参终点保存精确摘要和运行时有限性检查，首批额外保留数组，最终5K保留全部终点。未保留的调参终点不声称经过事后逐值复核。本轮扫描与最终输出均为完整5K，保留原始batch来源。所有FID从缓存特征用FP64重算并核对。这是同一特征上的算术检查，不是独立特征提取器。固定前6个ID展示，不按观感筛图。','',
        '[完整CSV（含IS、SiT sFID、耗时）](data/ig_sg_5k_20260914/results.csv) · [审计与机器结果](data/ig_sg_5k_20260914/results.json) · [冻结协议](GUIDANCE_5K_GRID_20260914_ZH.md) · [实现与队列](../experiments/ig_sg_5k_20260914/run.py)。','',f'完整原始数据：`{r.ROOT}`。']
    REPORT.write_text('\n'.join(lines)+'\n')
    result=dict(complete=True,rows=all_rows,audits=all_audits,comparisons=comparisons,invalid=invalid,
        final_new_quality_images=75000,new_images=sum(x['samples'] for x in all_rows if not x['reused']),
        reused_images=sum(x['samples'] for x in all_rows if x['reused']),max_fid_error=max(x['error'] for x in all_audits),report_sha256=c.sha(REPORT))
    c.atomic(OUT/'results.json',result);c.atomic(r.ROOT/'report_complete.json',dict(complete=True,report=str(REPORT),report_sha256=c.sha(REPORT)))
    print('Final SG 5K comparisons',comparisons,flush=True)

if __name__=='__main__':main()
