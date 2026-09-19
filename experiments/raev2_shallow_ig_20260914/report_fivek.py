"""Report the amended 5K coefficient search and fresh held-out 5K."""
import csv, shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from . import grid as f
from . import report as old
c,m,s=f.c,f.m,f.s
OUT=c.WORK/'docs/data/raev2_shallow_ig_5k_selection_20260914'
REPORT=c.WORK/'docs/RAEV2_SHALLOW_IG_5K_SELECTION_RESULTS_20260914_ZH.md'


def main():
    assert c.read(m.ROOT/'grid_pipeline_complete.json')['complete']
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    if not (m.ROOT/'report_complete.json').exists():old.main()
    original_out=old.OUT;old.OUT=OUT;reference=old.reference_moments();rows=[];audits=[]
    for stage in f.STAGES:
        a,b=old.audit(stage,reference);rows+=a;audits+=b
    old.OUT=original_out
    legacy=f.legacy_record()
    fields=['stage','arm','kind','w','samples','fid','inception_score','full_calls','prefix_calls','head_calls','seconds','source_output']
    with (OUT/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    chosen=c.read(m.ROOT/'grid_selected_coefficients.json');shutil.copy2(m.ROOT/'grid_selected_coefficients.json',OUT/'selected_coefficients.json')
    tuning=[r for r in rows if r['stage']!='final_5k_v2'];final=[r for r in rows if r['stage']=='final_5k_v2']
    selected=[next(r for r in final if r['kind']==kind and r['w']==v['best_w']) for kind,v in chosen.items()]
    native=next(r for r in selected if r['kind']=='incumbent')
    default=next(r for r in final if r['kind']=='incumbent' and r['w']==1.78)
    best=min([r for r in selected if r['kind']!='incumbent'],key=lambda x:x['fid'])
    for kind in s.FAMILIES:
        best_tune=min([r for r in tuning if r['kind']==kind or r['w']==1],key=lambda x:(x['fid'],x['w']))
        assert chosen[kind]['best_w']==best_tune['w'] and chosen[kind]['tuning_fid']==best_tune['fid']
    input_stamp=c.read(m.ROOT/'final_v2_inputs/complete.json');assert input_stamp['seed']==2026091451
    assert c.sha(m.ROOT/'final_v2_inputs/noise.npy')!=c.sha(s.LEGACY/'inputs/first.npy')
    coarse=tuning;fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for kind in s.FAMILIES:
        for ax,data in zip(axes,[coarse,tuning]):
            values=sorted([r for r in data if (r['kind']==kind or r['w']==1) and (ax is axes[0] or r['w']<=1.8)],key=lambda x:x['w'])
            ax.plot([r['w'] for r in values],[r['fid'] for r in values],'-o',label=old.NAMES[kind],markersize=4)
    for ax,n in zip(axes,[5000,5000]):ax.set(xlabel='Extrapolation coefficient w',ylabel=f'FID ({n} selection images)');ax.grid(alpha=.2)
    axes[0].set_title('Complete5K scan and refinement');axes[1].set_title('Zoom: w <= 1.8');axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'selection_curves.png',dpi=160);fig.savefig(OUT/'selection_curves.pdf');plt.close(fig)
    canvas=Image.new('RGB',(260+6*160,35+len(final)*190),'white');draw=ImageDraw.Draw(canvas)
    labels=np.load(m.ROOT/'final_v2_inputs/labels.npy');draw.text((8,8),'Fresh final5K: first six fixed IDs; no sample selection',fill='black')
    for i,row in enumerate(final):
        top=35+i*190;draw.text((8,top+10),old.NAMES[row['kind']],fill='black');draw.text((8,top+30),f"w={row['w']:.3f}; FID={row['fid']:.4f}",fill='black')
        with np.load(Path(row['source_output'])/'samples.npz') as d:images=d['arr_0'][:6]
        for j,pixel in enumerate(images):
            x=260+j*160;canvas.paste(Image.fromarray(pixel).resize((156,156)),(x,top));draw.text((x,top+157),f'ID{j}; class{labels[j]}',fill='black')
    canvas.save(OUT/'paired_final_5k.png')
    lines=['# RAEv2：50K 后训练、5K 外推系数选择、独立 5K 确认','',
        f"已完成。后训练头中最终最低 FID 为 **{old.NAMES[best['kind']]}，w={best['w']:.3f}，FID={best['fid']:.4f}**；相对重新选参的原生 IG 差值为 **{best['fid']-native['fid']:+.4f}**，相对默认 w=1.78 原生 IG 差值为 **{best['fid']-default['fid']:+.4f}**。负数表示改善；单个最终 5K bank 不足以把细小差异判为稳定收益。",'',
        '原生 IG 位于第 8 层。第 4 层同结构 DDTFinalLayer 和仓库 Context MLP 均完成 50K；既有第 8 层 Context MLP 从 20K 精确续训到 50K。强主干冻结，保留原在线／EMA／AdamW／随机状态及训练审计。','',
        '所有粗扫系数与局部细化点均直接使用全类别 5K，1K 排名不参与候选淘汰。每个头先在w=1至2按0.2扫描，再按两端5K FID均值选择前三个相邻区间，以0.05补齐内部点。共享w=1；默认原生w=1.78单独作对照，不用于网格区间排序。选定后用全新 seed2026091451 的 5K 确认，同时采样默认原生基线。两组 5K 均覆盖 1000 类、每类 5 张，同组所有设置配对。','',
        '| 设置 | 选定 w | 选参 5K FID | 独立最终 5K FID↓ | IS↑ | 对重新选参原生 | 对默认原生 |','|---|---:|---:|---:|---:|---:|---:|']
    for row in selected:
        lines.append(f"| {old.NAMES[row['kind']]} | {row['w']:.3f} | {chosen[row['kind']]['tuning_fid']:.4f} | {row['fid']:.4f} | {row['inception_score']:.3f} | {row['fid']-native['fid']:+.4f} | {row['fid']-default['fid']:+.4f} |")
    invalid=[dict(c.read(p),stage=stage) for stage in f.STAGES for p in sorted((m.ROOT/stage).glob('*/invalid.json'))]
    if invalid:
        lines+=['','5K扫描中的数值无效点：'+'；'.join(f"{x['config']['kind']} w={x['config']['w']:.4f}（{x['stage']}，{x['error']}）" for x in invalid)+'。整组不参与FID选择，不报部分样本指标。']
    lines+=['',f"默认原生 w=1.78 在这组新最终输入上的 FID 为 **{default['fid']:.4f}**。",'',
        '![5K 粗扫和细化](data/raev2_shallow_ig_5k_selection_20260914/selection_curves.png)','',
        '原方案的完整或部分5K批次，仅在新粗网格或实际选中的细扫点与其参数匹配时复用；未被选中细扫区间内的旧点不参与选参。默认原生同一bank的历史5K仅保留作参考。它们均不是本修订的独立最终测试；新最终样本全部重新生成。修订时已知的原生和第 4 层同结构头结果在补充协议中披露。','',
        'Euler100、shift8、[.1,1] 活动窗口、FP32 velocity、BF16 模型、batch16 与 Nanogen 参考保持一致；每图 100 次全主干、零额外前缀，新头在 w>1 时调用 99 次。已完成组的提前评价与其余采样共享 GPU0，墙钟时间不作为独占速度基准。','',
        '逐批核对输入、类别、覆盖、全部像素与实际调用数。调参终点保存精确摘要和生成时有限性检查记录，首批保留数组；最终5K保留全部终点数组。未保存的调参终点不声称经过事后逐值复核。所有FID从同一缓存特征用FP64重算；这验证算术，不能视为第二个独立评价器。有限扫描的最低 FID 不意味着连续全局最优。20K 与 50K 头在不同系数下的差异不能单独归因于训练时长。','',
        '![固定配对样本](data/raev2_shallow_ig_5k_selection_20260914/paired_final_5k.png)','',
        '[完整结果 CSV](data/raev2_shallow_ig_5k_selection_20260914/results.csv) · [选定系数](data/raev2_shallow_ig_5k_selection_20260914/selected_coefficients.json) · [5K 选参补充协议](GUIDANCE_5K_GRID_20260914_ZH.md) · [原阶段与训练审计](RAEV2_SHALLOW_IG_50K_RESULTS_20260914_ZH.md)。']
    REPORT.write_text('\n'.join(lines)+'\n')
    result=dict(complete=True,rows=rows,audits=audits,legacy=legacy,selected=chosen,invalid=invalid,final_new_images=sum(x['samples'] for x in final),
        max_fid_error=max(x['error'] for x in audits),report_sha256=c.sha(REPORT))
    c.atomic(OUT/'results.json',result);c.atomic(m.ROOT/'fivek_report_complete.json',dict(complete=True,report=str(REPORT),report_sha256=c.sha(REPORT)))
    print('Amended final5K',[(x['kind'],x['w'],x['fid']) for x in final],flush=True)

if __name__=='__main__':main()
