"""Audit new/reused paired samples and archive inspectable pilot evidence."""
import argparse,csv,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from experiments.ig_sg_20260914 import run as run
from experiments.guidance_pasted_20260912.audit import fid_from_features
c=run.c.common
EVIDENCE=c.WORK/'docs/data/ig_sg_20260914'


def archive(name,phase):
    root=run.c.ROOT/phase/name;request=run.verify(root)
    noise=np.load(root/'noise.npy',mmap_mode='r');labels=np.load(root/'labels.npy')
    classes=100 if name=='sit_small' else 1000
    assert np.array_equal(np.bincount(labels,minlength=classes),np.full(classes,1000//classes))
    with np.load(request['reference']) as d:
        mu,cov=(d['mu'],d['sigma']) if 'mu' in d else (d['ref_mu'],d['ref_sigma'])
    evidence=(EVIDENCE if phase=='screen_1k' else EVIDENCE/phase)/name;evidence.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(240+6*144,35+len(request['configs'])*165),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,8),f'{name}: first six fixed IDs, paired inputs; no example selection',fill='black')
    rows=[];audits=[]
    for i,cfg in enumerate(request['configs']):
        reuse=request['reused_arms'].get(cfg['arm']);out=Path(reuse['out']) if reuse else root/cfg['arm']
        if not (out/'metrics.json').exists():continue
        original_request=out.parent/'request.json';summary=c.read(out/'summary.json');metric=c.read(out/'metrics.json')[0]
        assert summary['complete'] and summary['samples']==1000
        assert c.sha(out/'samples.npz')==summary['samples_sha256']
        assert c.sha(original_request)==summary['request_sha256']
        if name!='sit_small':assert metric['sample_sha256']==summary['samples_sha256']
        with np.load(out/'samples.npz') as d:images=d['arr_0']
        assert images.shape==(1000,256,256,3) and images.dtype==np.uint8
        covered=[];seconds=0
        for record in summary['records']:
            path=Path(record['path']);assert c.sha(path)==record['sha256']
            with np.load(path) as d:
                start=int(d['start']);stop=start+len(d['pixels']);covered.extend(range(start,stop))
                assert np.array_equal(d['pixels'],images[start:stop])
                assert np.array_equal(d['labels'],labels[start:stop])
                assert str(d['noise_sha256'])==c.array_sha(noise[start:stop])
                assert str(d['request_sha256'])==c.sha(original_request)
                assert int(d['full'])==summary['full_calls_per_output']
                assert np.isfinite(d['latents']).all() and d['latents'].shape==(stop-start,*run.c.SHAPES[name])
                assert np.isfinite(d['trace']).all()
                seconds+=float(d['seconds'])
        assert covered==list(range(1000)) and abs(seconds-summary['seconds'])<1e-5
        stages=2 if cfg['solver']=='heun' else 1
        expected=stages*cfg['steps']*{'baseline':1,'sg':2,'log':3}[cfg['kind']]
        if cfg['kind']=='sg':expected-=1
        if cfg['kind']=='log' and stages==2:expected-=2
        assert summary['full_calls_per_output']==expected and summary['prefix_calls_per_output']==0
        if name=='sit_small':
            feature_path=out/'inception_activations.npz'
            with np.load(feature_path) as d:features=d['pool_3']
        else:
            paths=list((out/'features').glob('*.features.pt'));assert len(paths)==1;feature_path=paths[0]
            features=torch.load(feature_path,weights_only=True,map_location='cpu').numpy()
        assert len(features)==1000 and np.isfinite(features).all()
        fp64=fid_from_features(features,mu,cov);error=abs(fp64-metric['fid'])
        assert error<.002,(name,cfg['arm'],fp64,metric['fid'])
        audits.append(dict(arm=cfg['arm'],passed=True,coverage=1000,paired_inputs=True,samples_equal_batches=True,
            fid_fp64=fp64,fid_difference=error,independent_extractor=False,
            features_sha256=c.sha(feature_path),samples_sha256=summary['samples_sha256'],original_request_sha256=c.sha(original_request)))
        row={**cfg,'model':name,'samples':1000,'fid':metric['fid'],'sfid':metric.get('sfid'),
            'inception_score':metric['inception_score'],'full_calls_per_output':summary['full_calls_per_output'],
            'seconds':seconds,'reused':bool(reuse),'source_output':str(out)}
        rows.append(row);top=35+i*165
        draw.text((6,top+12),cfg['arm'],fill='black')
        draw.text((6,top+32),f"FID {row['fid']:.3f}; NFE {row['full_calls_per_output']}",fill='black')
        draw.text((6,top+52),'Prior run reused' if reuse else 'New generation',fill='black')
        for j,pix in enumerate(images[:6]):
            x=240+j*144;canvas.paste(Image.fromarray(pix).resize((140,140)),(x,top))
            draw.text((x,top+142),f'ID {j}, class {labels[j]}',fill='black')
        for source in ['summary.json','metrics.json']:
            shutil.copy2(out/source,evidence/f"{cfg['arm']}_{source}")
    by={r['arm']:r for r in rows}
    for row in rows:
        if 'ig' in by:
            row['delta_ig']=row['fid']-by['ig']['fid'];row['relative_seconds']=row['seconds']/by['ig']['seconds']
        control={'ig_sg_w1':'ig_sg_cost','ig_log_k02_w1':'ig_log_cost'}.get(row['arm'])
        if control in by:row['delta_cost_control']=row['fid']-by[control]['fid']
    canvas.save(evidence/'comparison.png');shutil.copy2(root/'request.json',evidence/'request.json')
    shutil.copy2(run.c.ROOT/'checks'/f'{name}.json',evidence/'implementation_check.json')
    result=dict(rows=rows,audits=audits,complete=len(rows)==len(request['configs']),raw_root=str(root),request_sha256=c.sha(root/'request.json'))
    c.atomic(evidence/'results.json',result)
    fields=['model','arm','kind','samples','fid','sfid','inception_score','delta_ig','delta_cost_control',
        'full_calls_per_output','seconds','relative_seconds','reused','source_output']
    with (evidence/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(name,len(rows),[(r['arm'],round(r['fid'],4)) for r in rows],flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',default='screen_1k');p.add_argument('--models',default='sit_small,jit,raev2')
    a=p.parse_args();torch.set_num_threads(2)
    results={name:archive(name,a.phase) for name in a.models.split(',')}
    dest=EVIDENCE if a.phase=='screen_1k' else EVIDENCE/a.phase
    # Carry forward already audited model results from this exact frozen phase.
    for model_root in (run.c.ROOT/a.phase).iterdir():
        cached=dest/model_root.name/'results.json'
        if model_root.name not in results and (model_root/'request.json').exists() and cached.exists():
            result=c.read(cached)
            assert result['request_sha256']==c.sha(model_root/'request.json')
            results[model_root.name]=result
    c.atomic(dest/'results.json',results)
    controller_path=run.c.ROOT/a.phase/'controller_complete.json'
    if all(r['complete'] for r in results.values()) and controller_path.exists():
        controller=c.read(controller_path);assert controller['complete']
        rows=[x for r in results.values() for x in r['rows']]
        c.atomic(dest/'audit.json',dict(complete=True,arms=len(rows),images=1000*len(rows),
            new_images=1000*sum(not x['reused'] for x in rows),reused_images=1000*sum(x['reused'] for x in rows),
            all_artifact_checks_passed=True,controller=controller,
            maximum_fid_difference=max(x['fid_difference'] for r in results.values() for x in r['audits'])))

if __name__=='__main__':main()
