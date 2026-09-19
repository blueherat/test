"""Audit each finished screen once and apply the prewritten promotion rules."""
import json
import os
import subprocess
import time
from . import common as c
from . import report

STAGES=['cfg_screen_400','ig_screen_400','ig_calibrated_screen_400']


def review():
    decisions=[]
    for model in c.MODELS:
        cfg=c.read(c.ROOT/model/STAGES[0]/'results.json')
        baseline=min((r for r in cfg if r['arm'].startswith('independent_')),key=lambda r:r['fid'])
        candidate=min((r for r in cfg if r['arm'].startswith('shared_')),key=lambda r:r['fid'])
        eligible=candidate['fid']<=baseline['fid']-2 and candidate['inception_score']>=.9*baseline['inception_score']
        decisions.append(dict(model=model,stage=STAGES[0],candidate=candidate['arm'],baseline=baseline['arm'],
            candidate_fid=candidate['fid'],baseline_fid=baseline['fid'],delta=candidate['fid']-baseline['fid'],eligible=eligible))
        raw={r['arm']:r for r in c.read(c.ROOT/model/STAGES[1]/'results.json')}
        for stage in STAGES[1:]:
            rows={r['arm']:r for r in c.read(c.ROOT/model/stage/'results.json')}
            candidate=rows['posterior'];names=['native_base','native_half','native_double','time_mean','permuted','reversed']
            controls=[rows[n] if n in rows else raw[n] for n in names]
            best=min(controls,key=lambda r:r['fid'])
            eligible=all(candidate['fid']<=r['fid']-2 for r in controls) and candidate['inception_score']>=.9*raw['native_base']['inception_score']
            decisions.append(dict(model=model,stage=stage,candidate='posterior',baseline=best['arm'],
                candidate_fid=candidate['fid'],baseline_fid=best['fid'],delta=candidate['fid']-best['fid'],eligible=eligible))
    c.atomic(c.ROOT/'screen_review.json',dict(passed=True,decisions=decisions,eligible=[r for r in decisions if r['eligible']],
        rule='Frozen 400-primary screen: FID at least 2 below prescribed control, IS at least 90% of native base; confirmation uses fresh noise.',
        not_a_statistical_significance_test=True))
    return decisions


def main():
    done=set()
    while len(done)<6:
        changed=False
        for model in c.MODELS:
            for stage in STAGES:
                key=(model,stage);root=c.ROOT/model/stage
                if key in done or not (root/'evaluation_complete.json').exists():continue
                log=(root/'final_audit.log').open('a')
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                subprocess.run([c.PYTHON,'-m','experiments.guidance_pasted_20260912.audit','--model',model,'--stage',stage],
                    env=env,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT,check=True);log.close()
                done.add(key);changed=True
        if changed:report.main()
        c.atomic(c.ROOT/'validation_status.json',dict(phase='auditing_screens',complete=len(done),total=6,pid=os.getpid(),audited=[list(k) for k in sorted(done)]))
        if len(done)<6:time.sleep(8)
    decisions=review();report.main()
    c.atomic(c.ROOT/'validation_status.json',dict(phase='screens_complete',complete=6,total=6,pid=os.getpid(),
        eligible=[r for r in decisions if r['eligible']],overall_research_goal_complete=False))
    print(json.dumps(decisions),flush=True)


if __name__=='__main__':main()
