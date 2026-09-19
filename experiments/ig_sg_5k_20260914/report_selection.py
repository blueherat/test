"""Report full5K SG selection and same-bank cost controls without held-out claims."""
import csv
from pathlib import Path
import numpy as np
import torch
from . import selection_only as run
from . import report as old
c,OUT,REPORT=run.c,old.OUT,old.REPORT


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    assert c.read(run.ROOT/'pipeline_complete.json')['independent_validation'] is False
    rows=[];audits=[];refs={}
    for phase in run.STAGES:run.execution(phase)
    for name in run.MODELS:
        path=run.f.verify('grid_tune',name)['reference']
        if path not in refs:refs[path]=old.reference(path)
        for phase in run.STAGES:
            assert c.read(run.ROOT/phase/'controller_complete.json')['complete']
            a,b=old.audit(phase,name,refs[path]);rows+=a;audits+=b
    fields=['phase','model','arm','kind','w','omega','steps','solver','samples','fid','sfid','inception_score','full_calls','seconds','reused','source_output']
    with (OUT/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# IG＋SG：完整 5K 系数扫描与成本对照','',
        '**已完成三个模型、两种 SG 的粗扫、细扫和同输入组的 NFE 成本对照。按用户要求取消独立 5K 验证。** 每个参与排名的点都是完整 5000 张。表中 SG 最佳值是在同一调参组上选出的最低 FID。','',
        '沿用此前冻结的 IG；RAEv2 为原生第8层、w=1.78。外推 v_out=v_IG+(w_SG−1)×(v_strong−v_reference)，只扫 SG 外推系数。原版 SG 噪声时刻偏移 0.01；log-score 同时间对称扰动 kappa=0.2。','',
        '| 模型 | IG FID | 原版 SG 最佳 w / FID↓ | 原版成本对照 | log-score 最佳 w / FID↓ | log-score 成本对照 |','|---|---:|---:|---:|---:|---:|']
    comparisons=[]
    for name in run.MODELS:
        values=[r for r in rows if r['model']==name];search=[r for r in values if r['phase']!='cost_5k']
        base=next(r for r in search if r['kind']=='baseline')
        best={kind:min([r for r in search if r['kind']==kind],key=lambda r:(r['fid'],r['w'])) for kind in ('sg','log')}
        chosen=c.read(run.ROOT/'selection'/f'{name}_grid_final.json')
        for kind in best:assert chosen[kind]['fid']==best[kind]['fid'] and chosen[kind]['w']==best[kind]['w']
        cost={r['arm']:r for r in values if r['phase']=='cost_5k'}
        sg,log=best['sg'],best['log'];sc,lc=cost['ig_sg_cost'],cost['ig_log_cost']
        lines.append(f"| {old.DISPLAY[name]} | {base['fid']:.4f} | {sg['w']:.2f} / {sg['fid']:.4f} | {sc['fid']:.4f} | {log['w']:.2f} / {log['fid']:.4f} | {lc['fid']:.4f} |")
        comparisons.append(dict(model=name,sg_w=sg['w'],log_w=log['w'],sg_delta_ig=sg['fid']-base['fid'],sg_delta_cost=sg['fid']-sc['fid'],log_delta_ig=log['fid']-base['fid'],log_delta_cost=log['fid']-lc['fid']))
        fig,ax=old.plt.subplots(figsize=(7,4))
        for kind,label in [('sg','Paper SG'),('log','Log-score SG')]:
            data=sorted([r for r in search if r['kind']==kind],key=lambda r:r['w'])
            ax.plot([r['w'] for r in data],[r['fid'] for r in data],'-o',label=label,markersize=3)
        ax.axhline(base['fid'],color='black',ls='--',label='IG only');ax.set(xlabel='SG extrapolation w',ylabel='FID, 5000 selection images',title=old.DISPLAY[name]);ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUT/f'{name}_search.png',dpi=160);old.plt.close(fig)
        show=[base,sg,log,sc,lc];fig,axes=old.plt.subplots(5,6,figsize=(12,10))
        labels=np.load(run.f.verify('grid_tune',name)['labels'])
        for i,row in enumerate(show):
            with np.load(Path(row['source_output'])/'samples.npz') as d:images=d['arr_0'][:6]
            for j,pixel in enumerate(images):
                axes[i,j].imshow(pixel);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
                if i==0:axes[i,j].set_title(f'ID {j}, class {labels[j]}',fontsize=8)
                if j==0:axes[i,j].set_ylabel(f"{row['arm']}\nFID={row['fid']:.4f}",fontsize=8)
        fig.suptitle('First six fixed paired IDs; selection bank and same-bank NFE controls')
        fig.tight_layout();fig.savefig(OUT/f'{name}_comparison.png',dpi=140);old.plt.close(fig)
    lines+=['','所有方法与成本对照均使用 seed2026091441 的同一配对 5K 输入组；SiT 100类每类50张，JiT／RAEv2 1000类每类5张。SiT 的 ADM ImageNet100 与另外两个模型的 Nanogen ImageNet256 参考不同，绝对 FID 不跨模型比较。','',
        'SiT 的 IG／原版 SG／log-score NFE 为128／255／382，成本对照为256／382；JiT 和 RAEv2 为100／199／300，成本对照为199／300。最优 SG 与原 IG 的已生成调参图像直接用于报告，不再另采新种子 5K。','',
        '粗扫范围 w=1–2、步长0.2，再对前三个完整端点均值最低的相邻区间以0.05细扫。数值无效点不计算部分 FID。“最佳”是正外推候选中的最低调参 FID；若它仍差于 IG，则不能称为收益。由于取消独立验证，所选最低值不能代表独立验证成绩。','',
        '逐批核对像素、输入、标签、覆盖与实际NFE；FID从同一缓存特征以FP64重算，属于算术复核。调参终点仅首批保留数组，其余保留运行时有限性记录和精确摘要。原版SG沿用发布代码的velocity外推；log-score版本不等同于密度高斯卷积后的精确score。','']
    for name in run.MODELS:lines += [f'![{old.DISPLAY[name]} 扫描](data/ig_sg_5k_20260914/{name}_search.png)','',f'![{old.DISPLAY[name]} 配对样本](data/ig_sg_5k_20260914/{name}_comparison.png)','']
    lines+=['[完整CSV](data/ig_sg_5k_20260914/results.csv) · [审计与机器结果](data/ig_sg_5k_20260914/results.json) · [执行修订](GUIDANCE_5K_SELECTION_ONLY_20260914_ZH.md)。']
    REPORT.write_text('\n'.join(lines)+'\n')
    invalid=[dict(c.read(p),path=str(p)) for phase in run.STAGES for p in (run.ROOT/phase).glob('*/*/invalid.json')]
    c.atomic(OUT/'results.json',dict(complete=True,rows=rows,audits=audits,comparisons=comparisons,invalid=invalid,
        independent_validation=False,final_new_quality_images=0,max_fid_error=max(x['error'] for x in audits),report_sha256=c.sha(REPORT)))
    c.atomic(run.ROOT/'report_complete.json',dict(complete=True,independent_validation=False,report=str(REPORT),report_sha256=c.sha(REPORT)))
    print('SG same-bank selection and cost-control report complete.',flush=True)

if __name__=='__main__':main()
