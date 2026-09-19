"""Final source-cohort provenance and unselected sample contact sheets."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from . import common as c
from . import report


def source_audit(model):
    root=c.ROOT/model/'ig_source_data'
    request=c.read(root/'request.json');request_hash=c.sha(root/'request.json')
    inputs=c.bank(model,'ig_source_data');ids={0:[],1:[]};seconds=0.;checks=[];times=None;substages=None
    for group in ('sources','assets','inputs'):
        for p,h in request[group].items():assert c.sha(p)==h,p
    for path in sorted((root/'features').glob('*.npz')):
        meta=c.read(path.with_suffix('.json'));assert c.sha(path)==meta['sha256']
        with np.load(path) as d:
            source=int(d['source']);index=d['ids'];ids[source].extend(index.tolist())
            assert str(d['request_sha256'])==request_hash
            np.testing.assert_array_equal(d['labels'],inputs[2][index])
            assert str(d['noise_sha256'])==c.array_sha(inputs[0][index])
            assert d['features'].dtype==np.float16 and np.isfinite(d['features']).all()
            if times is None:times=d['times'].copy();substages=d['substages'].copy()
            np.testing.assert_array_equal(d['times'],times);np.testing.assert_array_equal(d['substages'],substages)
            assert d['features'].shape[:2]==(len(index),len(times))
            seconds+=float(d['seconds'])
        checks.append(dict(file=str(path),sha256=meta['sha256']))
    for source,index in ids.items():assert sorted(index)==list(range(1200))
    for path in sorted((root/'features').glob('*_full.npz')):
        with np.load(path) as s,np.load(path.with_name(path.name.replace('_full','_base'))) as w:
            np.testing.assert_array_equal(s['features'][:,0],w['features'][:,0])
    complete=c.read(root/'complete.json');training=c.read(root/'training_complete.json')
    assert abs(seconds-complete['source_seconds'])<1e-7
    assert training['strong_gradients_absent'] and training['no_denoiser_training']
    assert training['head_sha256']==c.sha(root/'head.pt')
    calroot=c.ROOT/model/'ig_calibrated_source';cal=c.read(calroot/'request.json')
    assert cal['raw_request_sha256']==request_hash
    assert cal['raw_head_sha256']==c.sha(root/'head.pt')
    assert cal['head_sha256']==c.sha(calroot/'head.pt')
    for group in ('sources','inputs'):
        for path,h in cal[group].items():assert c.sha(path)==h,path
    raw=torch.load(root/'head.pt',map_location='cpu',weights_only=False)['state']
    corrected=torch.load(calroot/'head.pt',map_location='cpu',weights_only=False)['state']
    beta=cal['inverse_temperature']
    for name,value in raw.items():
        expected=value*beta if name in ('net.2.weight','net.2.bias') else value
        torch.testing.assert_close(corrected[name],expected,atol=0,rtol=0)
    row=dict(passed=True,model=model,source_feature_files=len(checks),source_seed_pairs=1200,
        train_seed_ids=[0,999],calibration_seed_ids=[1000,1099],audit_seed_ids=[1100,1199],
        split_by_whole_seed=True,previous_aggregate_validation_seen=True,
        query_rows_per_source=1200*len(times),same_initial_features_all_pairs_exact=True,
        source_seconds=seconds,head_training_seconds=training['train_seconds'],
        raw_head_sha256=training['head_sha256'],calibrated_head_sha256=cal['head_sha256'],
        temperature_only_change_exact=True,source_request_sha256=request_hash,feature_records=checks)
    c.atomic(root/'final_provenance.json',row)
    return {k:v for k,v in row.items() if k!='feature_records'}


def sheets():
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    records=[]
    for model in c.MODELS:
        for stage in report.STAGES:
            root=c.ROOT/model/stage
            if not (root/'evaluation_complete.json').exists():continue
            rows=c.read(root/'results.json');tile=160;label_width=255;top=66
            canvas=Image.new('RGB',(label_width+4*tile,top+len(rows)*(tile+8)),(255,255,255))
            draw=ImageDraw.Draw(canvas)
            draw.text((10,8),f'{model} / {stage}',font=font,fill='black')
            draw.text((10,31),'Fixed primary IDs 0, 1, 2, 3; no image selection. Columns paired only within this stage.',font=small,fill='black')
            samples=[]
            for index,row in enumerate(rows):
                p=root/row['arm']/'samples.npz'
                with np.load(p) as d:images=d['arr_0'][:4].copy()
                y=top+index*(tile+8)
                draw.text((10,y+35),report.EN[row['arm']],font=small,fill='black')
                draw.text((10,y+56),f"400-primary FID {row['fid']:.2f}",font=small,fill='black')
                for j,pixels in enumerate(images):
                    canvas.paste(Image.fromarray(pixels).resize((tile,tile),Image.Resampling.LANCZOS),(label_width+j*tile,y))
                samples.append(dict(arm=row['arm'],samples_sha256=c.sha(p),primary_ids=[0,1,2,3]))
            path=report.OUT/f'{model}_{stage}_first4.png';canvas.save(path)
            records.append(dict(model=model,stage=stage,file=str(path),sha256=c.sha(path),rows=samples))
    return records


def main():
    torch.set_num_threads(2)
    provenance=[source_audit(model) for model in c.MODELS]
    grids=sheets()
    stopped={}
    stop_expectations={
        'sit_refined_priority_20260911':'c8227c2a9aa9ccf49003bdb1652a4a6cce870fb79ae8066a7a475811e3bb3635',
        'sit_control_output_50ideas_20260910':'b43aef680652b33cc5a199cecb355a717ae50af5e058cecaa83d4f91aa180ce1',
        'sit_apg_mechanism_extension_20260911':'c8227c2a9aa9ccf49003bdb1652a4a6cce870fb79ae8066a7a475811e3bb3635',
        'sit_broad_resume_20260912':'c8227c2a9aa9ccf49003bdb1652a4a6cce870fb79ae8066a7a475811e3bb3635',
        'sit_reference_aggregation_20260912':'f17cc7a0cb994382cf856a190d0603caa6d452cbd62211c06075a16d60b7a09d'}
    for folder,expected in stop_expectations.items():
        p=c.ROOT.parent/folder/'STOP_AFTER_CURRENT';assert p.exists(),p
        actual=c.sha(p);assert actual==expected,p;stopped[str(p)]=actual
    c.atomic(report.OUT/'final_provenance.json',dict(passed=True,sources=provenance,sample_grids=grids,
        all_six_sample_grids_complete=len(grids)==6,old_stop_markers_unchanged=stopped))
    print(json.dumps(dict(passed=True,source_models=len(provenance),sample_grids=len(grids))))


if __name__=='__main__':main()
