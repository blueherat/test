"""Refine only the depth8 MLP from completed 5K coarse endpoints."""
import argparse,sys,time,subprocess
from pathlib import Path
from . import grid as f
c,m,s,b=f.c,f.m,f.s,f.b
MODULE='experiments.raev2_shallow_ig_20260914.mlp8_refine'
STAGE='mlp8_refine_5k'
PROTOCOL=c.WORK/'docs/RAEV2_MLP8_REFINE_ONLY_20260914_ZH.md'
SNAPSHOT=m.ROOT/'coarse_snapshot_mlp8_refine.json'


def snapshot():
    if SNAPSHOT.exists():
        result=c.read(SNAPSHOT)
        for p,h in result['assets'].items():assert c.sha(p)==h,p
        return result
    req=s.verify('grid_5k');root=m.ROOT/'grid_5k';rows=[];assets={str(root/'request.json'):c.sha(root/'request.json')};cancelled=[]
    for cfg in req['configs']:
        out=root/cfg['arm'];skip=out/'skipped.json'
        if skip.exists():cancelled.append(c.read(skip));assets[str(skip)]=c.sha(skip);continue
        assert (out/'metrics.json').exists(),cfg
        summary=c.read(out/'summary.json');assert summary['complete'] and summary['samples']==5000
        rows.append(dict(cfg,stage='grid_5k',fid=c.read(out/'metrics.json')[0]['fid']))
        for p in (out/'summary.json',out/'metrics.json'):assets[str(p)]=c.sha(p)
    assert any(x['kind']=='mlp8' and x['w']==1.8 for x in rows)
    result=dict(rows=rows,assets=assets,cancelled=cancelled,original_coarse_plan_complete=False,active_family='mlp8')
    c.atomic(SNAPSHOT,result);return result


def prepare():
    root=m.ROOT/STAGE
    if (root/'request.json').exists():req=s.verify(STAGE)
    else:
        saved=snapshot();intervals,values=f.g.best_intervals([x for x in saved['rows'] if x['kind']=='mlp8' or x['w']==1])
        detail=dict(kind='mlp8',intervals=intervals,fine_coefficients=values,samples=5000,independent_validation=False)
        path=m.ROOT/'mlp8_refinement_selection.json';c.atomic(path,detail)
        req=dict(s.verify('grid_5k'));req.update(stage=STAGE,configs=[f.config('mlp8',w) for w in values],prefix_records={},reused_arms={},
            sources=dict(req['sources']),selection={str(SNAPSHOT):c.sha(SNAPSHOT),str(path):c.sha(path),**saved['assets']},
            selection_rule='Only depth8 MLP; completed w1.8 retained, w2 stopped; top3 completed endpoint intervals, step0.05, each5K',independent_validation=False)
        for p in (Path(__file__).resolve(),PROTOCOL):req['sources'][str(p)]=c.sha(p)
        f.attach_reuse(req);c.atomic(root/'request.json',req)
    for cfg in req['configs']:f.copy_reuse(STAGE,cfg,req)
    c.atomic(root/'prepared.json',dict(complete=True,request_sha256=c.sha(root/'request.json')))
    print('Prepared MLP8-only refinement:',[x['w'] for x in req['configs']],'; reused:',list(req['reused_arms']),flush=True)
    return req


def early():
    while not (m.ROOT/STAGE/'prepared.json').exists():time.sleep(5)
    req=s.verify(STAGE)
    for cfg in req['configs']:
        out=m.ROOT/STAGE/cfg['arm']
        while not (out/'invalid.json').exists() and not all(b.record_path(req,cfg,i).exists() for i in range(0,5000,16)):time.sleep(5)
        if (out/'invalid.json').exists():continue
        f.locked_collect(STAGE,cfg,req);f.previous.locked_evaluate(STAGE,cfg,0)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','pipeline','worker','early_evaluate']);p.add_argument('--stage',default=STAGE,choices=[STAGE]);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    if a.action=='worker':s.save_npz=f.g.writer(True);f.previous.worker(STAGE,a.rank);return
    if a.action=='early_evaluate':early();return
    req=prepare()
    if a.action=='prepare':return
    b.MODULE=MODULE;b.collect=f.locked_collect;s.evaluate=f.previous.locked_evaluate
    done=m.ROOT/STAGE/'controller_complete.json'
    if not (done.exists() and c.read(done)['complete']):b.controller(STAGE)
    available=[x for x in snapshot()['rows'] if x['kind']=='mlp8' or x['w']==1]+b.rows((STAGE,))
    best=min(available,key=lambda x:(x['fid'],x['w']))
    c.atomic(m.ROOT/'mlp8_selected_coefficient.json',dict(best=best,independent_validation=False,samples=5000))
    c.atomic(m.ROOT/'mlp8_refine_complete.json',dict(complete=True,stage=STAGE,original_coarse_plan_complete=False,independent_validation=False))
    subprocess.run([sys.executable,'-u','-m','experiments.raev2_shallow_ig_20260914.report_mlp8'],check=True,cwd=c.WORK)

if __name__=='__main__':main()
