"""5K SG selection plus same-bank NFE controls; no independent confirmation."""
import argparse,subprocess,sys
from pathlib import Path
from . import grid as f
c,ROOT,MODELS,m,old=f.c,f.ROOT,f.MODELS,f.m,f.old
MODULE='experiments.ig_sg_5k_20260914.selection_only'
STAGES=('grid_tune','grid_refine','cost_5k')
PROTOCOL=c.WORK/'docs/GUIDANCE_5K_SELECTION_ONLY_20260914_ZH.md'


def sources():return {str(p):c.sha(p) for p in (Path(__file__).resolve(),PROTOCOL)}


def execution(phase):
    path=ROOT/phase/'execution_manifest.json'
    if path.exists():
        for p,h in c.read(path)['sources'].items():assert c.sha(p)==h,p
    else:c.atomic(path,dict(sources=sources(),requests={str(ROOT/phase/name/'request.json'):c.sha(ROOT/phase/name/'request.json') for name in MODELS},independent_validation=False,selection_samples=5000))
    for p,h in c.read(path)['requests'].items():assert c.sha(p)==h,p


def prepare(phase):
    assert phase in STAGES
    if phase!='cost_5k':
        existing={name:(ROOT/phase/name/'request.json').exists() for name in MODELS}
        f.prepare(phase)
        for name in MODELS:
            if existing[name]:continue
            path=ROOT/phase/name/'request.json';req=c.read(path)
            req.update(sources=dict(req['sources'],**sources()),independent_validation=False)
            c.atomic(path,req)
    else:
        for name in MODELS:
            path=ROOT/phase/name/'request.json'
            if path.exists():f.verify(phase,name);continue
            req=f.verify('grid_tune',name);base=m.configs(name)
            req=dict(req,phase=phase,configs=[base[3],base[4]],sources=dict(req['sources'],**sources()),
                selection_rule='Same full5K tuning inputs; two NFE controls, no independent seed or duplicate best-SG sampling',independent_validation=False)
            c.atomic(path,req)
            print('Prepared',phase,name,'two same-bank NFE controls x5000',flush=True)
    execution(phase)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pipeline','prepare','controller','worker','evaluate_one']);p.add_argument('--phase',choices=STAGES,default='grid_tune')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--model',choices=MODELS);p.add_argument('--arm');a=p.parse_args()
    if a.action=='prepare':prepare(a.phase)
    elif a.action=='worker':execution(a.phase);f.r.save_npz=f.g.writer(True);f.r.worker(a.phase,a.rank)
    elif a.action=='controller':execution(a.phase);f.r.MODULE=MODULE;f.r.collect=f.collect;f.r.controller(a.phase)
    elif a.action=='evaluate_one':old.evaluate(ROOT/a.phase/a.model/a.arm,a.model)
    else:
        for phase in STAGES:
            done=ROOT/phase/'controller_complete.json'
            if done.exists() and c.read(done)['complete']:continue
            for action in ('prepare','controller'):subprocess.run([sys.executable,'-u','-m',MODULE,action,'--phase',phase],check=True,cwd=c.WORK)
        for name in MODELS:f.choices('selection_only',name)
        c.atomic(ROOT/'pipeline_complete.json',dict(complete=True,models=3,stages=STAGES,independent_validation=False,final_new_samples=0,cost_control_samples=30000,sources=sources()))
        subprocess.run([sys.executable,'-u','-m','experiments.ig_sg_5k_20260914.report_selection'],check=True,cwd=c.WORK)

if __name__=='__main__':main()
