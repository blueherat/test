"""Held-pixel affine-color explanation of decoded head differences; not guidance."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.raev2_training_core import file_sha256

def main():
    root=Path('/home/zhoushunyu/data/eqvae/experiments/ig_head_correspondence_20260908/decoded')
    meta=json.loads((root/'results.json').read_text());assert meta['complete'] and meta['images']==96
    for p,h in meta['files'].items():assert file_sha256(root/p)==h
    yy,xx=np.indices((256,256));train=((yy//8+xx//8)%2==0).reshape(-1);rows=[]
    for i in [0,142,285,428,570,713,856,999]:
        for k in [0,47,73,87]:
            pix={n:np.asarray(Image.open(root/f'id{i:04d}_step{k:03d}_{n}.png')).reshape(-1,3).astype(float)/255 for n in ['Base','Full','IG_clean']}
            for source,target in [('Base','Full'),('Full','IG_clean')]:
                a,b=pix[source],pix[target];pred=np.empty_like(a);params=[]
                for c in range(3):
                    x=np.stack([a[:,c],np.ones(len(a))],axis=1)
                    theta=np.linalg.lstsq(x[train],b[train,c],rcond=None)[0]
                    pred[:,c]=x@theta;params.append(theta.tolist())
                original=float(np.square(a[~train]-b[~train]).mean())
                residual=float(np.square(pred[~train]-b[~train]).mean())
                rows.append(dict(id=i,step=k,source=source,target=target,original_mse=original,
                                 residual_mse=residual,explained_fraction=1-residual/max(original,1e-30),parameters=params))
    grouped=[]
    for k in [0,47,73,87]:
        for source in ['Base','Full']:
            rr=[r for r in rows if r['step']==k and r['source']==source]
            grouped.append(dict(step=k,source=source,held_pixel_explained_fraction=1-sum(r['residual_mse'] for r in rr)/max(sum(r['original_mse'] for r in rr),1e-30),
                                original_mse=float(np.mean([r['original_mse'] for r in rr]))))
    result=dict(complete=True,rows=rows,grouped=grouped,source_sha256=file_sha256(root/'results.json'),
                scope='6 affine color parameters per image; fit alternating 8x8 tiles, evaluate other tiles; clipped uint8 readouts; no image-quality inference')
    out=Path(__file__).resolve().parents[1]/'experiments/results/terminal_defect_20260908/raev2_head_photometric.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(grouped,indent=2))
if __name__=='__main__':main()
