"""Finish evaluation and reporting for retained 1.25/1.30/1.35 only; no sampling."""
import subprocess,sys
from pathlib import Path
from . import mlp8_refine as run
c,m,s,b,f=run.c,run.m,run.s,run.b,run.f
RETAINED=('mlp8_w1p2500','mlp8_w1p3000','mlp8_w1p3500')
REQUEST=m.ROOT/'finish135_request.json'


def main():
    amendment=c.read(REQUEST);root=m.ROOT/run.STAGE;req=s.verify(run.STAGE)
    assert amendment['retained_arms']==list(RETAINED) and amendment['independent_validation'] is False
    assert amendment['request_sha256']==c.sha(root/'request.json')
    configs=[x for x in req['configs'] if x['arm'] in RETAINED];assert len(configs)==3
    for cfg in configs:
        assert all(b.record_path(req,cfg,start).exists() for start in range(0,5000,16)),cfg
        f.locked_collect(run.STAGE,cfg,req);f.previous.locked_evaluate(run.STAGE,cfg,0)
        summary=c.read(root/cfg['arm']/'summary.json');assert summary['complete'] and summary['samples']==5000
    fine=[dict(cfg,stage=run.STAGE,fid=c.read(root/cfg['arm']/'metrics.json')[0]['fid']) for cfg in configs]
    available=[x for x in run.snapshot()['rows'] if x['kind']=='mlp8' or x['w']==1]+fine
    best=min(available,key=lambda x:(x['fid'],x['w']))
    c.atomic(m.ROOT/'mlp8_selected_coefficient.json',dict(best=best,independent_validation=False,samples=5000))
    c.atomic(m.ROOT/'mlp8_refine_complete.json',dict(complete=True,stage=run.STAGE,retained_arms=list(RETAINED),
        original_coarse_plan_complete=False,original_fine_plan_complete=False,independent_validation=False,
        amendment_sha256=c.sha(REQUEST),finisher_source_sha256=c.sha(Path(__file__).resolve())))
    c.atomic(root/'status.json',dict(complete=True,phase='retained_points_complete',arms=3,original_plan_complete=False))
    subprocess.run([sys.executable,'-u','-m','experiments.raev2_shallow_ig_20260914.report_mlp8'],check=True,cwd=c.WORK)

if __name__=='__main__':main()
