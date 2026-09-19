"""Read-only compact progress for head and SG5K queues."""
import json
from pathlib import Path
ROOT=Path('/home/zhoushunyu/data/eqvae/experiments')

def show(root,configs,n,batch):
    rows=[];cancelled=[]
    for cfg in configs:
        out=root/cfg['arm'];invalid=out/'invalid.json'
        if (out/'skipped.json').exists():cancelled.append(json.loads((out/'skipped.json').read_text()));continue
        if invalid.exists():rows.append(dict(arm=cfg['arm'],invalid=True));continue
        summary=out/'summary.json';metric=out/'metrics.json'
        count=json.loads(summary.read_text()).get('samples',n) if summary.exists() else sum(min(batch,n-int(p.stem[5:])) for p in out.glob('batch*.npz'))
        row=dict(arm=cfg['arm'],samples=count)
        if metric.exists():
            data=json.loads(metric.read_text());row['fid']=(data[0] if isinstance(data,list) else data)['fid'];row['metric_time']=metric.stat().st_mtime
        rows.append(row)
    print(json.dumps(dict(stage=str(root.relative_to(ROOT)),planned_arms=len(rows),complete_arms=sum(x.get('samples',0)==n for x in rows),
        evaluated_arms=sum('fid' in x for x in rows),invalid_arms=sum(x.get('invalid',False) for x in rows),
        cancelled_arms=[dict(arm=x['arm'],partial_samples=x['partial_samples']) for x in cancelled],samples=sum(x.get('samples',0) for x in rows),active=[x for x in rows if 0<x.get('samples',0)<n],pending_evaluation=[x['arm'] for x in rows if x.get('samples',0)==n and 'fid' not in x],recent=sorted([x for x in rows if 'fid' in x],key=lambda x:x['metric_time'],reverse=True)[:3]),ensure_ascii=False))

h=ROOT/'raev2_shallow_ig_20260914'
for stage in ('grid_5k','mlp8_refine_5k'):
    p=h/stage/'request.json'
    if p.exists():
        r=json.loads(p.read_text());show(p.parent,r['configs'],r['samples'],r['batch'])
g=ROOT/'ig_sg_5k_20260914'
for stage in ('grid_tune','grid_refine','cost_5k'):
    for name in ('sit_small','jit','raev2'):
        p=g/stage/name/'request.json'
        if p.exists():
            r=json.loads(p.read_text());show(p.parent,r['configs'],r['samples'],r['batch'])
