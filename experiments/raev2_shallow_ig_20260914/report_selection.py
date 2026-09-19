"""Audited same-bank selection report; independent validation cancelled by user."""
import csv
from pathlib import Path
import numpy as np
import torch
from . import selection_only as run
from . import report as old
from . import report_fivek as previous
c,m=run.c,run.m
OUT,REPORT=previous.OUT,previous.REPORT


def main():
    run.install();torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    assert c.read(m.ROOT/'grid_pipeline_complete.json')['independent_validation'] is False
    old.OUT=OUT;reference=old.reference_moments();rows=[];audits=[]
    for stage in run.STAGES:
        run.execution(stage);a,b=old.audit(stage,reference);rows+=a;audits+=b
    chosen=c.read(m.ROOT/'grid_selected_coefficients.json');legacy=run.f.legacy_record()
    selected={}
    for kind,value in chosen.items():
        best=min([r for r in rows if r['kind']==kind or r['w']==1],key=lambda r:(r['fid'],r['w']))
        assert best['w']==value['best_w'] and best['fid']==value['tuning_fid']
        selected[kind]=best
    with (OUT/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    c.atomic(OUT/'selected_coefficients.json',chosen)
    fig,ax=old.plt.subplots(figsize=(8,4.5))
    for kind in run.s.FAMILIES:
        values=sorted([r for r in rows if r['kind']==kind or r['w']==1],key=lambda r:r['w'])
        ax.plot([r['w'] for r in values],[r['fid'] for r in values],'-o',label=old.NAMES[kind],markersize=3)
    ax.axhline(legacy['fid'],ls='--',color='black',label='Native IG w=1.78, same bank')
    ax.set(xlabel='Extrapolation coefficient w',ylabel='FID, 5000 selection images');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'selection_curves.png',dpi=160);old.plt.close(fig)
    fig,axes=old.plt.subplots(4,6,figsize=(12,8));labels=np.load(run.verify('grid_5k')['labels'])
    for i,(kind,row) in enumerate(selected.items()):
        with np.load(Path(row['source_output'])/'samples.npz') as d:images=d['arr_0'][:6]
        for j,pixel in enumerate(images):
            axes[i,j].imshow(pixel);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(f'ID {j}, class {labels[j]}',fontsize=8)
            if j==0:axes[i,j].set_ylabel(f"{old.NAMES[kind]}\nw={row['w']:.2f}, FID={row['fid']:.4f}",fontsize=8)
    fig.suptitle('Selected configurations, first six fixed paired IDs; same selection bank')
    fig.tight_layout();fig.savefig(OUT/'paired_selection_5k.png',dpi=140);old.plt.close(fig)
    skipped=c.read(run.SKIP);native=selected['incumbent']
    lines=['# RAEv2：50K 后训练头、完整 5K 扫参与细化','',
        '**已完成保留的粗扫和细扫。按用户要求取消独立 5K 验证；下表是已测系数中的最低调参 FID。** 每个参与排名的点均为 5000 张，同组噪声和类别配对。','',
        '原生 IG 位于第8层。第4层 DDTFinalLayer、第4层 Context MLP 和第8层 Context MLP 均已训练到 50K；强主干冻结。粗扫步长 0.2，按完整端点 FID 均值选择前三个区间，内部步长 0.05。','',
        '| 头 | 选定 w | 调参 5K FID↓ | 相对原生最优 | 相对原生默认 |','|---|---:|---:|---:|---:|']
    for kind,row in selected.items():lines.append(f"| {old.NAMES[kind]} | {row['w']:.2f} | {row['fid']:.4f} | {row['fid']-native['fid']:+.4f} | {row['fid']-legacy['fid']:+.4f} |")
    lines+=['',f"原生默认 w=1.78 使用同一输入组的既有完整 5K，FID={legacy['fid']:.4f}；不重复生成。",'',
        f"第4层 MLP 的 w=2 按用户要求取消，保留 {skipped['partial_samples']} 张部分输出；不计算部分 FID，不视为数值失败，也不用于细扫区间排序。其他保留点继续按原协议执行。",'',
        '![系数扫描](data/raev2_shallow_ig_5k_selection_20260914/selection_curves.png)','',
        '![固定配对样本](data/raev2_shallow_ig_5k_selection_20260914/paired_selection_5k.png)','',
        '最低值是在同一个 5K 输入组上选择得到，不能称为独立验证收益，也不代表连续全局最优。Euler100、shift8、FP32 velocity、BF16 模型及原 Nanogen 参考保持一致。逐批检查输入、标签、像素、覆盖及调用数；FID 从同一缓存特征以 FP64 重算。调参终点仅首批保留数组，其余保存运行时有限性记录与精确摘要，不声称经过事后逐值检查。','',
        '[完整 CSV](data/raev2_shallow_ig_5k_selection_20260914/results.csv) · [审计与机器结果](data/raev2_shallow_ig_5k_selection_20260914/results.json) · [执行修订](GUIDANCE_5K_SELECTION_ONLY_20260914_ZH.md) · [训练审计](RAEV2_SHALLOW_IG_50K_RESULTS_20260914_ZH.md)。']
    REPORT.write_text('\n'.join(lines)+'\n')
    c.atomic(OUT/'results.json',dict(complete=True,rows=rows,audits=audits,selected=chosen,legacy=legacy,skipped=skipped,
        independent_validation=False,final_new_images=0,max_fid_error=max(x['error'] for x in audits),report_sha256=c.sha(REPORT)))
    c.atomic(m.ROOT/'fivek_report_complete.json',dict(complete=True,independent_validation=False,report=str(REPORT),report_sha256=c.sha(REPORT)))
    print('Head selection report complete; no independent validation.',flush=True)

if __name__=='__main__':main()
