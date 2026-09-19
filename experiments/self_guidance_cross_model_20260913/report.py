"""Archive paired metrics, inspect hashes, and independently recompute FID."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from PIL import Image,ImageDraw
from experiments.self_guidance_cross_model_20260913 import core as c
from experiments.self_guidance_cross_model_20260913.run import verify,REFERENCE
from experiments.guidance_pasted_20260912.audit import fid_from_features


def audit(out,request):
    summary=c.common.read(out/'summary.json');metric=c.common.read(out/'metrics.json')[0]
    assert metric['sample_sha256']==summary['samples_sha256']==c.common.sha(out/'samples.npz')
    assert summary['samples']==request['samples']
    with np.load(out/'samples.npz') as data:pixels=data['arr_0']
    noise=np.load(out.parent/'noise.npy',mmap_mode='r');labels=np.load(out.parent/'labels.npy')
    assert np.array_equal(np.bincount(labels,minlength=1000),np.full(1000,request['samples']//1000))
    covered=[];seconds=0.
    for record in summary['records']:
        path=Path(record['path']);assert c.common.sha(path)==record['sha256']
        with np.load(path) as d:
            start=int(d['start']);stop=start+len(d['pixels']);covered.extend(range(start,stop))
            assert np.array_equal(d['pixels'],pixels[start:stop])
            assert np.array_equal(d['labels'],labels[start:stop])
            assert str(d['noise_sha256'])==c.common.array_sha(noise[start:stop])
            assert str(d['request_sha256'])==c.common.sha(out.parent/'request.json')
            assert int(d['full'])==summary['full_calls_per_output']
            assert np.isfinite(d['latents']).all()
            seconds+=float(d['seconds'])
    assert covered==list(range(request['samples'])) and abs(seconds-summary['seconds'])<1e-6
    paths=list((out/'features').glob('*.features.pt'));assert len(paths)==1,paths
    features=torch.load(paths[0],map_location='cpu',weights_only=True).numpy()
    assert len(features)==request['samples'] and np.isfinite(features).all()
    with np.load(REFERENCE) as d:
        mu,cov=(d['mu'],d['sigma']) if 'mu' in d else (d['ref_mu'],d['ref_sigma'])
    computed=fid_from_features(features,mu,cov)
    error=abs(computed-metric['fid']);assert error<.002,(computed,metric['fid'])
    record=dict(passed=True,samples=request['samples'],fid_reported=metric['fid'],
        fid_fp64_same_features=computed,absolute_error=error,independent_feature_extractor=False,
        source_request_sha256=c.common.sha(out.parent/'request.json'),
        pixels_sha256=summary['samples_sha256'],features_sha256=c.common.sha(paths[0]),
        sample_coverage_labels_inputs_counts_timing=True)
    c.common.atomic(out/'audit.json',record)
    return summary,metric,record


def report(phase,name):
    root=c.ROOT/phase/name;request=verify(root)
    evidence=c.common.WORK/'docs/data/self_guidance_cross_model_20260913'/phase/name
    evidence.mkdir(parents=True,exist_ok=True)
    rows=[];audits=[]
    configs=[config for config in request['configs'] if (root/config['arm']/'metrics.json').exists()]
    if not configs:return
    canvas=Image.new('RGB',(190+6*144,30+len(configs)*165),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,8),f'{name} / {phase}: first 6 fixed IDs; paired noise and labels',fill='black')
    for index,config in enumerate(configs):
        out=root/config['arm'];summary,metric,record=audit(out,request)
        row={**config,'model':name,'samples':request['samples'],'fid':metric['fid'],
             'inception_score':metric['inception_score'],
             'full_calls_per_output':summary['full_calls_per_output'],'seconds':summary['seconds']}
        rows.append(row);audits.append(dict(arm=config['arm'],**record))
        y=30+index*165
        draw.text((6,y+20),config['arm'],fill='black')
        draw.text((6,y+40),f"FID {row['fid']:.3f} | NFE {row['full_calls_per_output']}",fill='black')
        with np.load(out/'batch000000.npz') as d:
            for j,pixel in enumerate(d['pixels'][:6]):
                x=190+j*144
                canvas.paste(Image.fromarray(pixel).resize((140,140)),(x,y))
                draw.text((x,y+142),f"ID {j}, class {d['labels'][j]}",fill='black')
    by={r['arm']:r for r in rows}
    for row in rows:
        if row['base'] in by:
            base=by[row['base']]
            row['delta_fid']=row['fid']-base['fid']
            row['relative_fid_percent']=100*row['delta_fid']/base['fid']
            row['relative_seconds']=row['seconds']/base['seconds']
    canvas.save(evidence/'comparison.png')
    fields=['model','arm','base','kind','omega','steps','samples','fid','delta_fid',
            'relative_fid_percent','inception_score','full_calls_per_output','seconds','relative_seconds']
    with (evidence/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    c.common.atomic(evidence/'results.json',dict(rows=rows,audits=audits,complete=len(rows)==len(request['configs']),
                                              raw_root=str(root),planned=len(request['configs'])))
    shutil.copy2(root/'request.json',evidence/'request.json')
    for path in (c.ROOT/'checks').glob(name+'*.json'):
        shutil.copy2(path,evidence/path.name)
    table=['|配置|FID ↓|ΔFID对同源基线|IS ↑|Full/图|采样+解码GPU秒|',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        delta=f"{r['delta_fid']:+.3f}" if 'delta_fid' in r else '—'
        table.append(f"|`{r['arm']}`|{r['fid']:.3f}|{delta}|{r['inception_score']:.3f}|{r['full_calls_per_output']}|{r['seconds']:.1f}|")
    (evidence/'table.md').write_text('\n'.join(table)+'\n')
    print(name,len(rows),'/',len(request['configs']));print('\n'.join(table))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='screen_1k');p.add_argument('--models',default='jit,raev2')
    a=p.parse_args();torch.set_num_threads(2)
    for name in a.models.split(','):report(a.phase,name)
    completion=c.ROOT/a.phase/'controller_complete.json'
    if completion.exists() and c.common.read(completion)['complete']:
        count=len(list((c.ROOT/a.phase).glob('*/*/metrics.json')))
        c.common.atomic(c.ROOT/a.phase/'status.json',dict(complete=True,completed=count,
            remaining_queue=0,exit_codes=c.common.read(completion)['exit_codes']))
