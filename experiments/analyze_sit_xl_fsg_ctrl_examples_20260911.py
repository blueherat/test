"""All fixed XL examples, with images shown at their original pixel size."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from experiments import run_sit_xl_fsg_ctrl_examples_20260911 as run
from experiments.analyze_sit_fsg_ctrl_hypothesis_20260911 import NAMES as SMALL_NAMES,font,class_names
from experiments.lifting_scale_sweep_20260909 import atomic,read,sha,WORK

OUT=WORK/'docs/data/sit_xl_fsg_ctrl_examples_20260911'
NAMES=dict(SMALL_NAMES,conditional='Conditional (a=0)',cfg_low='CFG a=0.5',smc01_high='CTRL K=0.1')
MAIN=('cfg_tuned','cfg_high','fsg_high','smc_high','instant_high')
FSG=('cfg_high','fsg_high','fsg_length_high','fsg_debiased_high','smc01_high','smc_high')


def load():
    rows=[];traces=[]
    request_hash=sha(run.ROOT/'request.json')
    for method in run.METHODS:
        for rank in range(4):
            path=run.ROOT/f'{method}_{rank}.json'
            if not path.exists():continue
            item=read(path);assert item['request_sha256']==request_hash
            assert sha(path.with_suffix('.npz'))==item['arrays_sha256']
            rows+=item['rows'];traces+=item['traces']
    return pd.DataFrame(rows),pd.DataFrame(traces)


def sheet(frame,methods=MAIN,suffix='guided',step=64,indices=(0,1,2,3),name='main'):
    with np.load(run.ROOT/'inputs.npz') as data:labels=data['labels']
    names=class_names();tile=256;pad=12;left=220;top=94;caption=42
    image=Image.new('RGB',(left+len(methods)*(tile+pad)+pad,top+len(indices)*(tile+caption+pad)+pad),'white')
    draw=ImageDraw.Draw(image)
    draw.text((12,8),'SiT-XL/2 · fixed same-noise examples',font=font(24),fill='#17212b')
    for j,method in enumerate(methods):
        label=NAMES[method].replace('FSG length / CFG direction','FSG length, CFG direction').replace('FSG minus same-field error','FSG minus cycle error')
        draw.text((left+j*(tile+pad),53),label,font=font(17),fill='#17212b')
    for i,index in enumerate(indices):
        y0=top+i*(tile+caption+pad)
        draw.text((12,y0+20),f'#{index:03d}',font=font(22),fill='#17212b')
        label=names[int(labels[index])];words=label.split();lines=['']
        for word in words:
            if len(lines[-1]+' '+word)>17:lines.append(word)
            else:lines[-1]=(lines[-1]+' '+word).strip()
        for n,line in enumerate(lines):draw.text((12,y0+58+24*n),line,font=font(18),fill='#17212b')
        for j,method in enumerate(methods):
            folder='guided' if suffix=='guided' else f'{suffix}_{step:02d}'
            path=run.ROOT/'images'/method/folder/f'{index:03d}.png'
            if not path.exists():continue
            x0=left+j*(tile+pad);pixels=Image.open(path).convert('RGB');assert pixels.size==(256,256)
            image.paste(pixels,(x0,y0))
            row=frame[(frame['index']==index)&(frame.method==method)&(frame.k==step)&(frame.suffix==suffix)]
            if len(row):
                r=row.iloc[0]
                draw.text((x0,y0+tile+5),f'P(target): R18 {r.resnet_p:.3f} / CN {r.convnext_p:.3f}',font=font(13),fill='#334155')
    path=OUT/f'{name}_{suffix}_{step:02d}.png';image.save(path);return path


def gallery(frame):
    with np.load(run.ROOT/'inputs.npz') as data:labels=data['labels']
    names=class_names()
    data=json.dumps(dict(samples=[dict(index=i,label=names[int(y)]) for i,y in enumerate(labels)],
        methods=list(run.METHODS),names=NAMES,
        rows={f'{r.method}/{r.suffix}/{r.k}/{r.index}':dict(r18=r.resnet_p,cn=r.convnext_p,r18ok=r.resnet_top1,cnok=r.convnext_top1) for r in frame.itertuples(index=False)}),ensure_ascii=False).replace('</','<\\/')
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SiT-XL FSG / CTRL 单图对照</title>
<style>body{font:15px system-ui,sans-serif;background:#f5f7fa;color:#1c2936;margin:24px}h1{font-size:25px}p{max-width:1100px;line-height:1.7}select{padding:10px;min-width:300px;border:1px solid #bac5d0;border-radius:6px;background:white}table{border-collapse:separate;border-spacing:10px}th{text-align:left;min-width:140px;vertical-align:top}td{background:white;border:1px solid #d6dee7;padding:10px;border-radius:5px}img{display:block;width:256px;height:256px}small{color:#526270;line-height:1.6}.scroll{overflow:auto}</style>
<h1>SiT-XL/2：FSG / CFG-Ctrl 单图对照</h1><p>固定采用小模型机制集前16个噪声及对应ImageNet-1K类别。每个方法的5种结局都保留。当前参数是机制迁移探针；16张图片不构成FID或普遍质量排名。R18 / CN是后验分类器的目标概率。</p>
<p>大模型使用官方IG checkpoint的full分支，真NULL类别为1000。时间统一为噪声0到图像1；后续纯条件与NULL是两种不同操作。点击图片查看原始像素。</p><label>样本与目标　<select id="sample"></select></label><div class="scroll" id="view"></div>
<script>const D=__DATA__,s=document.querySelector('#sample');D.samples.forEach(r=>{let o=document.createElement('option');o.value=r.index;o.textContent=`#${String(r.index).padStart(3,'0')} · ${r.label}`;s.append(o)});
function draw(){let i=+s.value,cols=[['guided',64,'继续原 guidance'],['null',16,'t=0.25 → NULL'],['null',32,'t=0.50 → NULL'],['null',48,'t=0.75 → NULL'],['conditional',32,'t=0.50 → 纯条件']],body='<tr><th>方法</th>'+cols.map(c=>`<th>${c[2]}</th>`).join('')+'</tr>';
for(let m of D.methods){body+=`<tr><th>${D.names[m]}</th>`;for(let [suffix,k] of cols){let r=D.rows[`${m}/${suffix}/${k}/${i}`],folder=suffix==='guided'?'guided':`${suffix}_${String(k).padStart(2,'0')}`,path=`images/${m}/${folder}/${String(i).padStart(3,'0')}.png`;body+='<td>'+(r?`<a href="${path}" target="_blank"><img loading="lazy" src="${path}" alt="同噪声生成图片"></a><small>R18 ${r.r18.toFixed(3)} · CN ${r.cn.toFixed(3)}<br>top1: ${r.r18ok?'✓':'×'} / ${r.cnok?'✓':'×'}</small>`:'尚未完成')+'</td>'}body+='</tr>'}document.querySelector('#view').innerHTML='<table>'+body+'</table>'}s.addEventListener('change',draw);draw();</script></html>'''
    (run.ROOT/'gallery.html').write_text(document.replace('__DATA__',data))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame,traces=load()
    assert not frame.duplicated(['method','k','suffix','index']).any()
    frame.to_csv(OUT/'image_readouts.csv',index=False)
    traces.to_csv(OUT/'coordinate_traces.csv',index=False)
    frame.groupby(['method','k','suffix'])[['resnet_p','resnet_top1','convnext_p','convnext_top1']].mean().reset_index().to_csv(OUT/'means.csv',index=False)
    gallery(frame)
    paths=[sheet(frame),sheet(frame,suffix='null',step=32),sheet(frame,FSG,name='fsg_controls')]
    complete=all((run.ROOT/f'rank{r}.json').exists() and read(run.ROOT/f'rank{r}.json')['complete'] for r in range(4))
    if complete:assert len(frame)==960,len(frame)
    atomic(OUT/'coverage.json',dict(complete=complete,image_records=len(frame),samples=16,
        methods=12,request_sha256=sha(run.ROOT/'request.json'),sheets=[str(x) for x in paths]))
    print(dict(complete=complete,image_records=len(frame),sheets=[str(x) for x in paths]),flush=True)


if __name__=='__main__':main()
