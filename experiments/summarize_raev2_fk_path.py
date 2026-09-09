"""Summarize frozen candidate outputs; never generate or reevaluate a baseline."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/ig_fk_path_fourcard_20260908')
OUT=Path('experiments/results/terminal_defect_20260908/fk_path')
BASE=ROOT.parent/'ig_condition_carrier_20260908/quality/native_ig/samples.npz'


def main():
    rows=[]; seconds=[]; memory=[]; starts=[]; candidates={}
    for path in sorted(ROOT.glob('rank*/batch*.npz')):
        with np.load(path) as b:
            start=int(b['labels'][0]); starts.append(start)
            seconds.append(float(b['seconds'])); memory.append(int(b['peak_memory_bytes']))
            for row in json.loads(str(b['diagnostics_json'])):
                rows.append(row)
            if start<8:
                for label,pixel in zip(b['labels'],b['arr_0']):candidates[int(label)]=pixel
    if not rows: return
    keys=['logweight_mean','particle_ess','logweight_difference','gradient_rms','correction_rms','native_rms']
    result=dict(batches=len(starts),samples=4*len(starts),sum_batch_gpu_seconds=sum(seconds),
                median_batch_seconds=float(np.median(seconds)),max_peak_memory_bytes=max(memory),
                diagnostics={k:dict(mean=float(np.mean([r[k] for r in rows])),
                                   median=float(np.median([r[k] for r in rows])),
                                   max=float(np.max([r[k] for r in rows]))) for k in keys},
                by_time=[])
    for t in sorted(set(r['t'] for r in rows),reverse=True):
        rr=[r for r in rows if r['t']==t]
        result['by_time'].append(dict(t=t,**{k:float(np.mean([r[k] for r in rr])) for k in keys}))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    if len(candidates)==8:
        with np.load(BASE) as b: reference=b['arr_0'][:8]
        canvas=Image.new('RGB',(532,8*276+30),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((10,8),'Existing native IG',fill='black'); draw.text((276,8),'Direct stochastic FK',fill='black')
        for i in range(8):
            y=30+i*276
            canvas.paste(Image.fromarray(reference[i]),(0,y));canvas.paste(Image.fromarray(candidates[i]),(276,y))
            draw.text((2,y+256),f'label {i}',fill='black')
        canvas.save(OUT/'fixed_first8.png')
    print(json.dumps({k:v for k,v in result.items() if k!='by_time'},indent=2))

if __name__=='__main__':main()
