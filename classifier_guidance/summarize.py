"""Export measured performance and paired numerical checks to compact evidence."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance/performance_20260919')
OUT = Path(__file__).resolve().parents[1]/'docs/classifier_guidance'
PAIRS = (
    ('SiT', 'train_baseline', 'train_graph_checkpoint', .001),
    ('JiT speed', 'jit_train_baseline', 'jit_train_optimized', .005),
    ('JiT memory', 'jit_train_baseline', 'jit_train_low_memory', .005),
    ('RAEv2', 'raev2_train_baseline', 'raev2_train_optimized', .01),
)


def compare(a,b):
    a,b=a.astype(np.float64),b.astype(np.float64)
    return dict(relative_error=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
                max_error=float(np.abs(a-b).max()))


def main():
    rows=[];evidence=[]
    for name,old,new,tolerance in PAIRS:
        a=json.loads((ROOT/old/'result.json').read_text())
        b=json.loads((ROOT/new/'result.json').read_text())
        assert a['complete'] and b['complete'] and a['batch']==b['batch']
        checks={key:compare(np.load(ROOT/old/(key+'.npy')),np.load(ROOT/new/(key+'.npy')))
                for key in ('head_gradient','head_after','critic_after')}
        assert checks['head_gradient']['relative_error']<tolerance
        assert checks['head_gradient']['cosine']>.9999
        assert checks['critic_after']['max_error']==0
        assert a['iterations'][-1]['metrics'][:3]==b['iterations'][-1]['metrics'][:3]
        row=dict(model=name,batch=a['batch'],baseline_seconds=a['median_seconds'],optimized_seconds=b['median_seconds'],
            speedup=a['median_seconds']/b['median_seconds'],
            baseline_allocated_gib=a['peak_allocated_gib'],optimized_allocated_gib=b['peak_allocated_gib'],
            baseline_reserved_gib=a['peak_reserved_gib'],optimized_reserved_gib=b['peak_reserved_gib'],
            reserved_reduction=1-b['peak_reserved_gib']/a['peak_reserved_gib'],
            gradient_relative_error=checks['head_gradient']['relative_error'],gradient_cosine=checks['head_gradient']['cosine'])
        rows.append(row);evidence.append(dict(name=name,baseline=a,optimized=b,checks=checks,
                                             paths=dict(baseline=str(ROOT/old),optimized=str(ROOT/new))))
    smoke={}
    for name in ('sit_ddp_smoke','sit_resume_smoke','jit_train_smoke','raev2_train_smoke'):
        run=ROOT/name
        complete=json.loads((run/'complete.json').read_text())
        latest=json.loads((run/'latest.json').read_text())
        exit_status=json.loads((run/'exit.json').read_text())
        assert complete['complete'] and latest['replicas_identical'] and exit_status['exit_code']==0
        smoke[name]=dict(complete=complete,latest=latest,exit=exit_status,path=str(run))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'performance_20260919.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (OUT/'performance_20260919.json').write_text(json.dumps(dict(rows=rows,evidence=evidence,training_checks=smoke),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
