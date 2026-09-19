"""All5K: coarse step0.2, best three intervals refined at step0.05."""
import argparse,fcntl,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from . import fivek as previous
from . import grid_common as g
s,c,m,b=previous.s,previous.c,previous.m,previous.b
MODULE='experiments.raev2_shallow_ig_20260914.grid'
STAGES=('grid_5k','grid_refine_5k','final_5k_v2')
PROTOCOL=c.WORK/'docs/GUIDANCE_5K_GRID_20260914_ZH.md'
legacy_record=previous.legacy_record


def config(kind,w):
    cfg=s.config(kind,w)
    # Preserve the exact scalar in an existing equivalent request when reusing that point.
    for stage in ('confirm_5k','candidate_5k'):
        path=m.ROOT/stage/'request.json'
        if not path.exists():continue
        for old in c.read(path)['configs']:
            if old['arm']==cfg['arm']:
                assert old['kind']==kind and np.float32(old['w']-1)==np.float32(w-1)
                return old
    return cfg


def copy_reuse(stage,cfg,req):
    origin=req.get('reused_arms',{}).get(cfg['arm'])
    if not origin:return
    out=m.ROOT/stage/cfg['arm'];out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return
    source=Path(origin['out']);summary=c.read(source/'summary.json')
    for path,digest in origin['assets'].items():assert c.sha(path)==digest
    for name in ('samples.npz','metrics.json'):shutil.copy2(source/name,out/name)
    shutil.copytree(source/'features',out/'features',dirs_exist_ok=True)
    summary=dict(summary,request_sha256=c.sha(m.ROOT/stage/'request.json'),reused=True,new_samples=0,reused_prefix_samples=5000,
        original_source=origin,records=[dict(row,source_request_sha256=row.get('source_request_sha256',origin['request_sha256']),reused=True) for row in summary['records']])
    c.atomic(out/'summary.json',summary)


def attach_reuse(req):
    req['prefix_records']={};req['reused_arms']={}
    for parent_stage in ('confirm_5k','candidate_5k'):
        path=m.ROOT/parent_stage/'request.json'
        if not path.exists():continue
        parent=s.verify(parent_stage);rh=c.sha(path)
        assert parent['samples']==5000 and parent['inputs']==req['inputs']
        by={x['arm']:x for x in parent['configs']}
        req['selection'][str(path)]=rh
        for cfg in req['configs']:
            if cfg['arm'] not in by or cfg['arm'] in req['reused_arms']:continue
            old=by[cfg['arm']];assert cfg==old
            out=path.parent/cfg['arm'];records=req['prefix_records'].setdefault(cfg['arm'],{})
            for start in range(0,5000,16):
                source=b.record_path(parent,cfg,start)
                if not source.exists() or str(start) in records:continue
                origin=parent.get('prefix_records',{}).get(cfg['arm'],{}).get(str(start));source_rh=origin['source_request_sha256'] if origin else rh
                with np.load(source) as data:assert int(data['start'])==start and str(data['request_sha256'])==source_rh
                records[str(start)]=dict(path=str(source),sha256=c.sha(source),source_request_sha256=source_rh)
            if (out/'metrics.json').exists() and (out/'summary.json').exists():
                summary=c.read(out/'summary.json');assert summary['samples']==5000 and summary['complete']
                assets={str(p):c.sha(p) for p in [out/'summary.json',out/'metrics.json',out/'samples.npz',*sorted((out/'features').glob('*'))] if p.is_file()}
                req['reused_arms'][cfg['arm']]=dict(out=str(out),request_sha256=rh,assets=assets)
                req['selection'].update(assets)


def prepare(stage):
    root=m.ROOT/stage;root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists():
        req=s.verify(stage)
        for cfg in req['configs']:copy_reuse(stage,cfg,req)
        if not (root/'prepared.json').exists():c.atomic(root/'prepared.json',dict(complete=True,request_sha256=c.sha(root/'request.json')))
        return req
    req=dict(s.verify('confirm_5k'));req['sources']=dict(req['sources']);req['selection']={}
    req['prefix_records']={};req['reused_arms']={}
    if stage=='grid_5k':
        cfgs=[config('incumbent',w) for w in g.COARSE]+[config(k,w) for k in s.FAMILIES[1:] for w in g.COARSE if w>1]
        detail=dict(coarse_step=.2,grid=list(g.COARSE),samples=5000,uses_1k_rankings=False)
        selection=m.ROOT/'grid_coarse_selection.json'
    elif stage=='grid_refine_5k':
        rows=b.rows(('grid_5k',));cfgs=[];detail={}
        for kind in s.FAMILIES:
            selected,values=g.best_intervals([row for row in rows if row['kind']==kind or row['w']==1])
            detail[kind]=dict(intervals=selected,fine_coefficients=values)
            cfgs.extend(config(kind,w) for w in values)
        selection=m.ROOT/'grid_refinement_selection.json'
    else:
        rows=b.rows(STAGES[:2]);detail={}
        for kind in s.FAMILIES:
            best=min([row for row in rows if row['kind']==kind or row['w']==1],key=lambda row:(row['fid'],row['w']))
            detail[kind]=dict(best_w=best['w'],tuning_fid=best['fid'],selection_stage=best['stage'],selection_arm=best['arm'])
        cfgs=[config(kind,value['best_w']) for kind,value in detail.items()]+[config('incumbent',1.78)]
        selection=m.ROOT/'grid_selected_coefficients.json'
    c.atomic(selection,detail);req['selection'][str(selection)]=c.sha(selection)
    paths=previous.final_bank() if stage=='final_5k_v2' else s.bank('confirm_5k')
    req.update(stage=stage,samples=5000,configs=cfgs,noise=str(paths[0]),labels=str(paths[1]),inputs={str(p):c.sha(p) for p in paths},
        selection_rule='Coarsew1..2 step0.2, top3 adjacent intervals by mean endpoint5K FID, interior step0.05; independent final5K',
        storage='All pixels and features retained. Tuning endpoints retain exact SHA256/finite check and first batch; final endpoints all retained.')
    for p in (Path(__file__).resolve(),Path(previous.__file__).resolve(),Path(g.__file__).resolve(),PROTOCOL):req['sources'][str(p)]=c.sha(p)
    if stage!='grid_5k':
        for parent_stage in (('grid_5k',) if stage=='grid_refine_5k' else STAGES[:2]):
            parent=m.ROOT/parent_stage
            for p in [parent/'request.json',*sorted(parent.glob('*/metrics.json')),*sorted(parent.glob('*/invalid.json'))]:req['selection'][str(p)]=c.sha(p)
    if stage!='final_5k_v2':attach_reuse(req)
    c.atomic(root/'request.json',req)
    for cfg in cfgs:copy_reuse(stage,cfg,req)
    c.atomic(root/'prepared.json',dict(complete=True,request_sha256=c.sha(root/'request.json')))
    print('Prepared',stage,len(cfgs),'arms x5000',flush=True);return req


def collect(stage,cfg,req):
    out=m.ROOT/stage/cfg['arm']
    if (out/'summary.json').exists() or (out/'invalid.json').exists():return
    rh=c.sha(m.ROOT/stage/'request.json');noise=np.load(req['noise'],mmap_mode='r');labels=np.load(req['labels'])
    images=[];records=[];seconds=0.;coverage=[];reused=0
    for start in range(0,5000,16):
        path=b.record_path(req,cfg,start);origin=req.get('prefix_records',{}).get(cfg['arm'],{}).get(str(start));source_rh=origin['source_request_sha256'] if origin else rh
        digest=c.sha(path)
        if origin:assert digest==origin['sha256']
        with np.load(path) as d:
            stop=start+len(d['pixels']);coverage.extend(range(start,stop))
            assert int(d['start'])==start and str(d['request_sha256'])==source_rh
            assert str(d['noise_sha256'])==c.array_sha(noise[start:stop]) and np.array_equal(d['labels'],labels[start:stop])
            g.check_latents(d,(stop-start,1024,16,16));images.append(d['pixels']);seconds+=float(d['seconds'])
            if origin:reused+=stop-start
        records.append(dict(path=str(path),sha256=digest,source_request_sha256=source_rh,reused=bool(origin)))
    assert coverage==list(range(5000));g.raw_save(out/'samples.npz',arr_0=np.concatenate(images))
    c.atomic(out/'summary.json',dict(complete=True,samples=5000,reused_prefix_samples=reused,new_samples=5000-reused,
        seconds=seconds,full_calls_per_output=100,prefix_calls_per_output=0,records=records,request_sha256=rh,samples_sha256=c.sha(out/'samples.npz')))


def locked_collect(stage,cfg,req):
    out=m.ROOT/stage/cfg['arm'];out.mkdir(parents=True,exist_ok=True)
    with (out/'evaluation.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX);return collect(stage,cfg,req)


def early():
    for stage in STAGES:
        while not (m.ROOT/stage/'prepared.json').exists():time.sleep(5)
        req=s.verify(stage)
        for cfg in req['configs']:
            out=m.ROOT/stage/cfg['arm']
            while not (out/'invalid.json').exists() and not all(b.record_path(req,cfg,i).exists() for i in range(0,5000,16)):time.sleep(5)
            if (out/'invalid.json').exists():continue
            locked_collect(stage,cfg,req);previous.locked_evaluate(stage,cfg,0)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pipeline','prepare','worker','early_evaluate']);p.add_argument('--stage');p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    if a.action=='worker':s.save_npz=g.writer(a.stage!='final_5k_v2');previous.worker(a.stage,a.rank);return
    if a.action=='early_evaluate':early();return
    if a.action=='prepare':prepare(a.stage);return
    b.MODULE=MODULE;b.collect=locked_collect;s.evaluate=previous.locked_evaluate
    for stage in STAGES:
        done=m.ROOT/stage/'controller_complete.json'
        if done.exists() and c.read(done)['complete']:continue
        prepare(stage);b.controller(stage)
    c.atomic(m.ROOT/'grid_pipeline_complete.json',dict(complete=True,stages=STAGES,coarse_step=.2,fine_step=.05,selection_samples=5000,final_samples=5000))
    subprocess.run([sys.executable,'-u','-m','experiments.raev2_shallow_ig_20260914.report_fivek'],check=True,cwd=c.WORK,
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))

if __name__=='__main__':main()
