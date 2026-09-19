"""Audit actual artifacts, archive paired contact sheets, recompute FP64 FID."""
from pathlib import Path
import argparse,csv,json,shutil
import numpy as np
import torch
from PIL import Image,ImageDraw
from experiments.guidance_pasted_20260912 import common as c
from experiments.cfg_transport_search_20260913 import runner as sit
from experiments.weak_reference_20260914 import cross_run as cross
from experiments.guidance_pasted_20260912.audit import fid_from_features

ROOT=c.EXPS/'weak_reference_20260914'
EVIDENCE=c.WORK/'docs/data/weak_reference_20260914'


def archive(phase,model):
    root=ROOT/phase if model=='sit' else ROOT/phase/model
    request=(sit.verify if model=='sit' else cross.verify)(root)
    reference=sit.REFERENCE if model=='sit' else cross.REFERENCE
    with np.load(reference) as ref:
        mu,cov=(ref['mu'],ref['sigma']) if 'mu' in ref else (ref['ref_mu'],ref['ref_sigma'])
    if model=='sit':
        with np.load(root/'inputs.npz') as bank:noise=bank['noise'];labels=bank['labels']
    else:
        noise=np.load(root/'noise.npy',mmap_mode='r');labels=np.load(root/'labels.npy')
    classes=100 if model=='sit' else 1000
    assert np.array_equal(np.bincount(labels,minlength=classes),np.full(classes,request['samples']//classes))
    evidence=EVIDENCE/phase/model;evidence.mkdir(parents=True,exist_ok=True)
    configs=[cfg for cfg in request['configs'] if (root/cfg['arm']/('fid.json' if model=='sit' else 'metrics.json')).exists()]
    canvas=Image.new('RGB',(240+6*144,30+len(configs)*165),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,8),f'{phase}/{model}: first six fixed IDs, paired inputs; no example selection',fill='black')
    rows=[];audits=[]
    for i,cfg in enumerate(configs):
        out=root/cfg['arm'];summary=c.read(out/'summary.json')
        metric=c.read(out/'fid.json') if model=='sit' else c.read(out/'metrics.json')[0]
        assert c.sha(out/'samples.npz')==summary['samples_sha256']
        assert summary['request_sha256']==c.sha(root/'request.json')
        assert summary['samples']==request['samples']
        covered=[];seconds=0
        for record in summary['records']:
            path=Path(record['path']);assert c.sha(path)==record['sha256']
            with np.load(path) as d:
                start=int(d['start']);stop=start+len(d['pixels']);covered.extend(range(start,stop))
                assert np.array_equal(d['labels'],labels[start:stop])
                assert str(d['noise_sha256'])==c.array_sha(noise[start:stop])
                assert str(d['request_sha256'])==c.sha(root/'request.json')
                assert int(d['full'])==summary['full_calls_per_output'] and np.isfinite(d['latents']).all()
                assert d['pixels'].dtype==np.uint8 and d['pixels'].shape==(stop-start,256,256,3)
                seconds+=float(d['seconds'])
        assert covered==list(range(request['samples'])) and abs(seconds-summary['seconds'])<1e-5
        if model=='sit':
            feature_path=out/'inception_activations.npz'
            with np.load(feature_path) as d:features=d['pool_3'].copy()
        else:
            paths=list((out/'features').glob('*.features.pt'));assert len(paths)==1
            feature_path=paths[0];features=torch.load(feature_path,weights_only=True,map_location='cpu').numpy()
        assert len(features)==request['samples'] and np.isfinite(features).all()
        computed=fid_from_features(features,mu,cov)
        error=abs(computed-metric['fid']);assert error<.002,(cfg['arm'],computed,metric['fid'])
        audits.append(dict(arm=cfg['arm'],passed=True,fid_fp64=computed,fid_difference=error,
            independent_extractor=False,features_sha256=c.sha(feature_path),samples_sha256=summary['samples_sha256']))
        row={**cfg,'model':model,'samples':request['samples'],'fid':metric['fid'],
            'sfid':metric.get('sfid'),'inception_score':metric['inception_score'],
            'full_calls_per_output':summary['full_calls_per_output'],'seconds':seconds}
        rows.append(row);top=30+i*165
        draw.text((6,top+15),cfg['arm'],fill='black')
        draw.text((6,top+35),f"FID {row['fid']:.3f}; NFE {row['full_calls_per_output']}",fill='black')
        with np.load(out/'batch000000.npz') as d:
            for j,pixel in enumerate(d['pixels'][:6]):
                x=240+j*144;canvas.paste(Image.fromarray(pixel).resize((140,140)),(x,top))
                draw.text((x,top+142),f'ID {j}; class {labels[j]}',fill='black')
    by={r['arm']:r for r in rows}
    for row in rows:
        base=('cfg_e64' if row.get('alpha') else 'strong_e64') if model=='sit' else row['base']
        if base in by:
            row['baseline']=base;row['delta_fid']=row['fid']-by[base]['fid']
            row['relative_seconds']=row['seconds']/by[base]['seconds']
    canvas.save(evidence/'comparison.png')
    c.atomic(evidence/'results.json',dict(rows=rows,audits=audits,complete=len(rows)==len(request['configs']),
        raw_root=str(root),request_sha256=c.sha(root/'request.json')))
    shutil.copy2(root/'request.json',evidence/'request.json')
    fields=['model','arm','kind','samples','fid','sfid','inception_score','baseline','delta_fid','full_calls_per_output','seconds','relative_seconds']
    with (evidence/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(model,phase,len(rows),'/',len(request['configs']),flush=True)
    for row in rows:print(row['arm'],round(row['fid'],4),row['full_calls_per_output'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--models',default='sit')
    args=p.parse_args();torch.set_num_threads(2)
    for model in args.models.split(','):archive(args.phase,model)
