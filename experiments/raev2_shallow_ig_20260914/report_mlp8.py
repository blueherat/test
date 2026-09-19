"""Audit retained full5K runs and depth8 MLP refinement after scope reduction."""
import csv
from pathlib import Path
import numpy as np
import torch
from . import mlp8_refine as run
from . import report as old
from . import report_fivek as previous
c,m,OUT,REPORT=run.c,run.m,previous.OUT,previous.REPORT


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);old.OUT=OUT
    assert c.read(m.ROOT/'mlp8_refine_complete.json')['complete']
    saved=run.snapshot();req=run.s.verify('grid_5k');arms={x['arm'] for x in saved['rows']}
    configs=[x for x in req['configs'] if x['arm'] in arms];ref=old.reference_moments()
    coarse,a=old.audit('grid_5k',ref,configs=configs,require_controller_complete=False)
    amendment_path=m.ROOT/'finish135_request.json'
    amendment=c.read(amendment_path) if amendment_path.exists() else None
    fine_req=run.s.verify(run.STAGE)
    if amendment:
        assert amendment['request_sha256']==c.sha(m.ROOT/run.STAGE/'request.json')
        fine_configs=[x for x in fine_req['configs'] if x['arm'] in amendment['retained_arms']]
        assert len(fine_configs)==3
        fine,b=old.audit(run.STAGE,ref,configs=fine_configs,require_controller_complete=False)
    else:fine,b=old.audit(run.STAGE,ref)
    rows=coarse+fine;audits=a+b
    best=min([x for x in rows if x['kind']=='mlp8' or x['w']==1],key=lambda x:(x['fid'],x['w']))
    chosen=c.read(m.ROOT/'mlp8_selected_coefficient.json')['best'];assert best['fid']==chosen['fid'] and best['w']==chosen['w']
    legacy=run.f.legacy_record();base=next(x for x in coarse if x['w']==1)
    with (OUT/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    c.atomic(OUT/'selected_coefficients.json',dict(mlp8=chosen,other_heads='further search abandoned by user'))
    fig,ax=old.plt.subplots(figsize=(8,4.5));data=sorted([x for x in rows if x['kind']=='mlp8' or x['w']==1],key=lambda x:x['w'])
    ax.plot([x['w'] for x in data],[x['fid'] for x in data],'-o',label='MLP depth8, 50K')
    ax.axhline(legacy['fid'],ls='--',color='black',label='Native IG w=1.78');ax.axhline(base['fid'],ls=':',label='No extrapolation')
    ax.set(xlabel='Extrapolation coefficient w',ylabel='FID, 5000 paired selection images');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'selection_curves.png',dpi=160);old.plt.close(fig)
    fig,axes=old.plt.subplots(1,6,figsize=(12,2.5));labels=np.load(req['labels'])
    with np.load(Path(best['source_output'])/'samples.npz') as d:images=d['arr_0'][:6]
    for j,pixel in enumerate(images):axes[j].imshow(pixel);axes[j].axis('off');axes[j].set_title(f'ID {j}, class {labels[j]}',fontsize=8)
    fig.suptitle(f"Selected MLP8 w={best['w']:.2f}, selection FID={best['fid']:.4f}; first six fixed IDs")
    fig.tight_layout();fig.savefig(OUT/'paired_selection_5k.png',dpi=140);old.plt.close(fig)
    lines=['# RAEv2：第8层 MLP 的 5K 细扫结果','',
        f"**已完成用户保留的第8层 MLP 细扫。已测点中最低调参 FID 为 {best['fid']:.4f}，w={best['w']:.2f}。** 第4层后续搜索放弃；独立验证已取消。",'',
        '| 设置 | w | 同组5K FID↓ |','|---|---:|---:|',f"| 不外推 | 1 | {base['fid']:.4f} |",f"| 原生默认 IG | 1.78 | {legacy['fid']:.4f} |",f"| 第8层 MLP，50K，已测最低 | {best['w']:.2f} | {best['fid']:.4f} |",'',
        f"第8层 MLP 相对原生默认 IG 的 FID 差为 {best['fid']-legacy['fid']:+.4f}。负数表示改善；这是同一调参组上的选择结果，不是独立验证成绩。",'',
        ('第8层 MLP 的 w=1.8 已完整跑完，FID=7.3673，w=2已停止。用户进一步要求只完成w=1.35；因此细扫保留已完成的w=1.25、复用w=1.30和补完的w=1.35，各5000张。w=1.45、1.50、1.55、1.65、1.70、1.75全部取消，不生成或计算部分FID。' if amendment else '第8层 MLP 的 w=1.8 已完整跑完，FID=7.3673。停止随后 w=2 的部分生成。仅在已完成粗扫中端点均值最低的三个区间 [1.2,1.4]、[1.4,1.6]、[1.6,1.8] 按0.05补齐九个内部点，每点5000张；同配置既有 w=1.30 完整5K经来源验证后复用，其余八点新生成。'),'',
        '![第8层 MLP 扫描](data/raev2_shallow_ig_5k_selection_20260914/selection_curves.png)','',
        '![固定前六个ID](data/raev2_shallow_ig_5k_selection_20260914/paired_selection_5k.png)','',
        '原有粗扫计划因用户调整而停止，不标记为原计划全部完成。已完成的其他头结果留在CSV中作参考；没有对第4层或原生IG做新的细扫。原生默认对照复用同一输入组的既有5K。','',
        '所有报告点均核对完整5000张、输入类别与噪声、像素和原始批次来源及实际调用数；FID从相同缓存特征以FP64重算。调参终点首批保留数组，其余保留精确摘要和运行时有限性检查，不声称经过事后逐值复核。','',
        '[完整 CSV](data/raev2_shallow_ig_5k_selection_20260914/results.csv) · [审计结果](data/raev2_shallow_ig_5k_selection_20260914/results.json) · [最新执行修订](RAEV2_MLP8_FINISH_135_20260914_ZH.md)。']
    REPORT.write_text('\n'.join(lines)+'\n')
    c.atomic(OUT/'results.json',dict(complete=True,rows=rows,audits=audits,selected=dict(mlp8=chosen),legacy=legacy,cancelled=saved['cancelled'],fine_amendment=amendment,original_coarse_plan_complete=False,original_fine_plan_complete=amendment is None,independent_validation=False,max_fid_error=max(x['error'] for x in audits),report_sha256=c.sha(REPORT)))
    c.atomic(m.ROOT/'fivek_report_complete.json',dict(complete=True,scope='mlp8_refinement_only',independent_validation=False,report=str(REPORT),report_sha256=c.sha(REPORT)))
    print('MLP8 refinement report complete:',best['w'],best['fid'],flush=True)

if __name__=='__main__':main()
