"""User amendment: cancel current MLP4 w2, retain 5K search, omit held-out runs."""
import argparse,sys,time,subprocess
from pathlib import Path
from . import grid as f
c,m,s,b=f.c,f.m,f.s,f.b
MODULE='experiments.raev2_shallow_ig_20260914.selection_only'
STAGES=f.STAGES[:2]
PROTOCOL=c.WORK/'docs/GUIDANCE_5K_SELECTION_ONLY_20260914_ZH.md'
SKIP=m.ROOT/'grid_5k/mlp4_w2p0000/skipped.json'
_verify=s.verify


def verify(stage):
    req=_verify(stage)
    if stage=='grid_5k':
        skip=c.read(SKIP)
        assert skip['cancelled_by_user'] and skip['request_sha256']==c.sha(m.ROOT/stage/'request.json')
        req=dict(req,configs=[x for x in req['configs'] if x['arm']!=skip['arm']])
    return req


def rows(stages):
    result=[]
    for stage in stages:
        assert c.read(m.ROOT/stage/'controller_complete.json')['complete']
        for cfg in verify(stage)['configs']:
            out=m.ROOT/stage/cfg['arm']
            if (out/'invalid.json').exists():continue
            assert c.read(out/'summary.json')['samples']==5000
            result.append(dict(cfg,stage=stage,fid=c.read(out/'metrics.json')[0]['fid']))
    return result


def sources():return {str(p):c.sha(p) for p in (Path(__file__).resolve(),PROTOCOL,SKIP)}


def execution(stage):
    path=m.ROOT/stage/'execution_manifest.json'
    if path.exists():
        manifest=c.read(path)
        for p,h in manifest['sources'].items():assert c.sha(p)==h,p
        assert manifest['request_sha256']==c.sha(path.parent/'request.json')
    else:
        req=verify(stage)
        c.atomic(path,dict(request_sha256=c.sha(path.parent/'request.json'),sources=sources(),
            active_arms=[x['arm'] for x in req['configs']],cancelled_arms=['mlp4_w2p0000'] if stage=='grid_5k' else [],
            selection_samples=5000,independent_validation=False))


def prepare(stage):
    assert stage in STAGES
    root=m.ROOT/stage
    if (root/'request.json').exists():
        req=verify(stage)
        for cfg in req['configs']:f.copy_reuse(stage,cfg,req)
    else:
        assert stage=='grid_refine_5k'
        available=rows(('grid_5k',));configs=[];detail={}
        for kind in s.FAMILIES:
            intervals,values=f.g.best_intervals([x for x in available if x['kind']==kind or x['w']==1])
            detail[kind]=dict(intervals=intervals,fine_coefficients=values)
            configs.extend(f.config(kind,w) for w in values)
        selected=m.ROOT/'grid_refinement_selection.json';c.atomic(selected,detail)
        req=dict(_verify('grid_5k'),stage=stage,configs=configs,prefix_records={},reused_arms={},
            selection={str(selected):c.sha(selected)},sources=dict(_verify('grid_5k')['sources'],**sources()),
            selection_rule='Full5K coarse and top3 valid endpoint intervals, fine0.05; user cancelled MLP4w2 and independent validation')
        parent=m.ROOT/'grid_5k'
        for p in [parent/'request.json',parent/'execution_manifest.json',SKIP,*sorted(parent.glob('*/metrics.json')),*sorted(parent.glob('*/invalid.json'))]:req['selection'][str(p)]=c.sha(p)
        f.attach_reuse(req);c.atomic(root/'request.json',req)
        for cfg in configs:f.copy_reuse(stage,cfg,req)
    execution(stage)
    c.atomic(root/'prepared.json',dict(complete=True,request_sha256=c.sha(root/'request.json')))
    print('Prepared selection-only',stage,len(req['configs']),'arms x5000',flush=True)
    return req


def install():s.verify=verify;b.rows=rows;b.MODULE=MODULE;b.collect=f.locked_collect;s.evaluate=f.previous.locked_evaluate


def early():
    for stage in STAGES:
        while not (m.ROOT/stage/'execution_manifest.json').exists():time.sleep(5)
        req=verify(stage)
        for cfg in req['configs']:
            out=m.ROOT/stage/cfg['arm']
            while not (out/'invalid.json').exists() and not all(b.record_path(req,cfg,i).exists() for i in range(0,5000,16)):time.sleep(5)
            if (out/'invalid.json').exists():continue
            f.locked_collect(stage,cfg,req);f.previous.locked_evaluate(stage,cfg,0)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pipeline','worker','early_evaluate']);p.add_argument('--stage',choices=STAGES);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    install()
    if a.action=='worker':
        execution(a.stage);s.save_npz=f.g.writer(True);f.previous.worker(a.stage,a.rank);return
    if a.action=='early_evaluate':early();return
    for stage in STAGES:
        done=m.ROOT/stage/'controller_complete.json'
        if done.exists() and c.read(done)['complete']:continue
        prepare(stage);b.controller(stage)
    available=rows(STAGES);chosen={}
    for kind in s.FAMILIES:
        best=min([r for r in available if r['kind']==kind or r['w']==1],key=lambda r:(r['fid'],r['w']))
        chosen[kind]=dict(best_w=best['w'],tuning_fid=best['fid'],selection_stage=best['stage'],selection_arm=best['arm'])
    c.atomic(m.ROOT/'grid_selected_coefficients.json',chosen)
    c.atomic(m.ROOT/'grid_pipeline_complete.json',dict(complete=True,stages=STAGES,selection_samples=5000,final_samples=0,independent_validation=False,cancelled=[str(SKIP)],sources=sources()))
    subprocess.run([sys.executable,'-u','-m','experiments.raev2_shallow_ig_20260914.report_selection'],check=True,cwd=c.WORK)

if __name__=='__main__':main()
