"""5K candidate selection amendment, followed by a fresh locked 5K bank."""
import argparse, fcntl, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from experiments.raev2_shallow_ig_20260914 import balanced as b
s, c, m = b.s, b.c, b.m
MODULE = 'experiments.raev2_shallow_ig_20260914.fivek'
STAGES = ('candidate_5k', 'refine_5k', 'final_5k_v2')
GRID = (1.,1.1,1.2,(1.2+1.4)/2,1.4,1.6,1.78,2.,2.4,3.)
PROTOCOL = c.WORK/'docs/RAEV2_SHALLOW_IG_5K_SELECTION_20260914_ZH.md'


def candidates():
    # Every grid point receives 5K, including points previously invalid on 1K.
    configs=[s.config('incumbent',w) for w in GRID]+[s.config(kind,w) for kind in s.FAMILIES[1:] for w in GRID if w>1]
    configs=[cfg for cfg in configs if not (cfg['kind']=='incumbent' and cfg['w']==1.78)]
    return configs,dict(grid=list(GRID),rule='All coefficients receive5K; no1K-based exclusion; native1.78 reused on exact same5K bank')


def final_bank():
    root = m.ROOT/'final_v2_inputs'; root.mkdir(parents=True, exist_ok=True)
    stamp = root/'complete.json'
    if not stamp.exists():
        rng = np.random.default_rng(2026091451); path=root/'noise.npy'
        x=np.lib.format.open_memmap(path.with_suffix('.tmp'), mode='w+', dtype=np.float32, shape=(5000,1024,16,16))
        for start in range(0,5000,8): x[start:start+8]=rng.standard_normal((8,1024,16,16),dtype=np.float32)
        x.flush(); del x; path.with_suffix('.tmp').replace(path)
        labels=np.arange(5000,dtype=np.int64)%1000; rng.shuffle(labels); np.save(root/'labels.npy',labels)
        c.atomic(stamp,dict(complete=True, seed=2026091451, samples=5000,
            inputs={str(root/f'{k}.npy'):c.sha(root/f'{k}.npy') for k in ('noise','labels')}))
    for p,h in c.read(stamp)['inputs'].items(): assert c.sha(p)==h
    assert np.array_equal(np.bincount(np.load(root/'labels.npy')),np.full(1000,5))
    return root/'noise.npy',root/'labels.npy'


def legacy_record():
    out=s.LEGACY/'native_base'
    files=[s.LEGACY/'request.json',out/'summary.json',out/'metrics.json',out/'audit.json',out/'samples.npz']
    files += sorted((out/'features').glob('*.pt'))
    audit=c.read(out/'audit.json'); metric=c.read(out/'metrics.json')
    assert audit['passed'] and audit['samples']==5000 and metric['request_sha256']==c.sha(s.LEGACY/'request.json')
    assert audit['samples_sha256']==metric['samples_sha256']==c.sha(out/'samples.npz')
    return dict(config=s.config('incumbent',1.78), fid=metric['fid'], source=str(out),
                assets={str(p):c.sha(p) for p in files}, audit=audit, reused=True)


def reuse_arm(stage,cfg,req):
    origin=req.get('reused_arms',{}).get(cfg['arm'])
    if not origin:return
    out=m.ROOT/stage/cfg['arm']; out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():return
    source=Path(origin['out']); summary=c.read(source/'summary.json')
    for p,h in origin['assets'].items():assert c.sha(p)==h
    shutil.copy2(source/'samples.npz',out/'samples.npz'); shutil.copy2(source/'metrics.json',out/'metrics.json')
    shutil.copytree(source/'features',out/'features',dirs_exist_ok=True)
    summary=dict(summary, request_sha256=c.sha(m.ROOT/stage/'request.json'), reused=True,
        new_samples=0, reused_prefix_samples=5000, original_source=origin,
        records=[dict(r,source_request_sha256=origin['request_sha256'],reused=True) for r in summary['records']])
    c.atomic(out/'summary.json',summary)


def prepare(stage):
    root=m.ROOT/stage; root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists():
        req=s.verify(stage)
        for cfg in req['configs']:reuse_arm(stage,cfg,req)
        if not (root/'prepared.json').exists():c.atomic(root/'prepared.json',dict(complete=True,request_sha256=c.sha(root/'request.json')))
        return req
    req=dict(s.verify('confirm_5k')); req['sources']=dict(req['sources']); req['selection']={}
    if stage=='candidate_5k':
        assert c.read(m.ROOT/'pipeline_complete.json')['complete']
        cfgs,ranking=candidates(); legacy=legacy_record()
        detail=dict(rule='Every coefficient on full grid receives5K; no exclusion from1K rankings',
            grid=ranking,legacy_native=legacy,reclassification='Original confirm_5k is now selection data; independent final uses seed2026091451')
        path=m.ROOT/'candidate_5k_selection.json'; c.atomic(path,detail)
        paths=s.bank('confirm_5k'); req['reused_arms']={}; req['prefix_records']={}
        previous=c.read(m.ROOT/'confirm_5k/request.json'); previous_rh=c.sha(m.ROOT/'confirm_5k/request.json')
        old_arms={cfg['arm'] for cfg in previous['configs']}
        for cfg in cfgs:
            if cfg['arm'] not in old_arms:continue
            out=m.ROOT/'confirm_5k'/cfg['arm']; summary=c.read(out/'summary.json')
            assert summary['complete'] and summary['samples']==5000
            assets={str(p):c.sha(p) for p in [out/'summary.json',out/'metrics.json',out/'samples.npz',*sorted((out/'features').glob('*'))] if p.is_file()}
            req['reused_arms'][cfg['arm']]=dict(out=str(out),request_sha256=previous_rh,assets=assets)
            req['prefix_records'][cfg['arm']]={str(int(Path(r['path']).stem[5:])):dict(r,source_request_sha256=previous_rh) for r in summary['records']}
            req['selection'].update(assets)
        req['selection'].update(legacy['assets'])
    elif stage=='refine_5k':
        rows=b.rows(('candidate_5k',));legacy=c.read(m.ROOT/'candidate_5k_selection.json')['legacy_native']
        rows.append(dict(legacy['config'],fid=legacy['fid'],stage='historical_same_bank_native'))
        detail={};cfgs=[]
        for kind in s.FAMILIES:
            best=min([r for r in rows if r['kind']==kind or r['w']==1],key=lambda r:(r['fid'],r['w']))
            w=best['w'];i=GRID.index(w);values=[]
            if i:values.append((w+GRID[i-1])/2)
            values.append((w+GRID[i+1])/2 if i+1<len(GRID) else 4.)
            detail[kind]=dict(best_coarse=best,refine_coefficients=values)
            cfgs.extend(s.config(kind,v) for v in values)
        path=m.ROOT/'refinement_5k_selection.json';c.atomic(path,detail)
        paths=s.bank('confirm_5k');req['reused_arms']={};req['prefix_records']={}
    else:
        assert c.read(m.ROOT/'refine_5k/controller_complete.json')['complete']
        rows=b.rows(STAGES[:2]); legacy=c.read(m.ROOT/'candidate_5k_selection.json')['legacy_native']
        rows.append(dict(legacy['config'],fid=legacy['fid'],stage='historical_same_bank_native'))
        detail={}
        for kind in s.FAMILIES:
            best=min([r for r in rows if r['kind']==kind or r['w']==1],key=lambda r:(r['fid'],r['w']))
            detail[kind]=dict(best_w=best['w'],tuning_fid=best['fid'],selection_stage=best['stage'],selection_arm=best['arm'])
        path=m.ROOT/'selected_coefficients_5k.json'; c.atomic(path,detail)
        cfgs=[s.config(kind,v['best_w']) for kind,v in detail.items()]
        if not any(x['kind']=='incumbent' and x['w']==1.78 for x in cfgs):cfgs.append(s.config('incumbent',1.78))
        paths=final_bank(); req['reused_arms']={}; req['prefix_records']={}
    req['selection'][str(path)]=c.sha(path)
    if stage!='candidate_5k':
        for prior_stage in (('candidate_5k',) if stage=='refine_5k' else STAGES[:2]):
            parent=m.ROOT/prior_stage
            for asset in [parent/'request.json',*sorted(parent.glob('*/metrics.json')),*sorted(parent.glob('*/invalid.json'))]:
                req['selection'][str(asset)]=c.sha(asset)
    req.update(stage=stage,samples=5000,configs=cfgs,noise=str(paths[0]),labels=str(paths[1]),inputs={str(p):c.sha(p) for p in paths},
        selection_rule='All coarse coefficients and local refinements use paired5K; fresh seed2026091451 final5K after lock')
    for p in (Path(__file__).resolve(),PROTOCOL):req['sources'][str(p)]=c.sha(p)
    c.atomic(root/'request.json',req)
    for cfg in cfgs:reuse_arm(stage,cfg,req)
    c.atomic(root/'prepared.json',dict(complete=True,request_sha256=c.sha(root/'request.json')))
    print('Prepared',stage,len(cfgs),flush=True);return req


@torch.inference_mode()
def worker(stage,rank):
    req=s.verify(stage); rt=c.runtime('raev2'); heads=s.load_heads(rt)
    captures={d:m.Capture(rt,d) for d in (4,8)}; calls=m.Counts(rt,heads)
    noise=np.load(req['noise'],mmap_mode='r'); labels=np.load(req['labels']); rh=c.sha(m.ROOT/stage/'request.json'); warmed=set()
    while (job:=s.pop(stage)) is not None:
        cfg=job['config']; out=m.ROOT/stage/cfg['arm']; out.mkdir(parents=True,exist_ok=True); start=job['start']
        if (out/'invalid.json').exists():continue
        try:
            if cfg['arm'] not in warmed:
                s.trajectory(rt,heads,captures,c.cuda(noise[:2]),c.cuda(labels[:2]),cfg); warmed.add(cfg['arm'])
            for start in range(job['start'],job['stop'],16):
                if (out/'invalid.json').exists():break
                if b.record_path(req,cfg,start).exists():continue
                stop=min(start+16,5000); x,y=c.cuda(noise[start:stop]),c.cuda(labels[start:stop])
                calls.reset(); torch.cuda.synchronize();begin=time.perf_counter()
                z,n=s.trajectory(rt,heads,captures,x,y,cfg); pixels=rt.decode(z);torch.cuda.synchronize();seconds=time.perf_counter()-begin
                expected=99 if cfg['kind']!='incumbent' and cfg['w']!=1 else 0
                assert n==dict(full=100,prefix=0) and calls.blocks==[100]*30 and sum(calls.heads.values())==expected
                s.save_npz(out/f'batch{start:06d}.npz',pixels=pixels,latents=z.cpu().numpy(),labels=labels[start:stop],start=start,
                    seconds=seconds,full=100,head_calls=expected,block_calls=np.array(calls.blocks),request_sha256=rh,noise_sha256=c.array_sha(noise[start:stop]))
                c.atomic(m.ROOT/stage/f'progress{rank}.json',dict(arm=cfg['arm'],start=start,stop=stop,phase='sampling'))
        except FloatingPointError as error:
            if stage=='final_5k_v2':raise
            with (out/'failure.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                failure=dict(invalid=True,config=cfg,start=start,error=str(error),request_sha256=rh,reason='Whole coefficient invalid')
                c.atomic(out/f'failure_rank{rank}_start{start:06d}.json',failure)
                if not (out/'invalid.json').exists():c.atomic(out/'invalid.json',failure)
    for cap in captures.values():cap.close()
    calls.close()


_BASE_COLLECT=b.collect
_BASE_EVALUATE=s.evaluate


def locked_collect(stage,cfg,req):
    out=m.ROOT/stage/cfg['arm'];out.mkdir(parents=True,exist_ok=True)
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);return _BASE_COLLECT(stage,cfg,req)


def locked_evaluate(stage,cfg,gpu):
    out=m.ROOT/stage/cfg['arm'];out.mkdir(parents=True,exist_ok=True)
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);return _BASE_EVALUATE(stage,cfg,gpu)


def early_evaluate():
    for stage in STAGES:
        while not (m.ROOT/stage/'prepared.json').exists():time.sleep(5)
        req=s.verify(stage)
        for cfg in req['configs']:
            out=m.ROOT/stage/cfg['arm']
            while not (out/'invalid.json').exists() and not all(b.record_path(req,cfg,i).exists() for i in range(0,5000,16)):time.sleep(5)
            if (out/'invalid.json').exists():continue
            locked_collect(stage,cfg,req);locked_evaluate(stage,cfg,0)
    c.atomic(m.ROOT/'fivek_early_evaluation_complete.json',dict(complete=True,note='Complete arms evaluated during other sampling; per-arm locks prevent duplicate collection/evaluation.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pipeline','worker','prepare','early_evaluate']);p.add_argument('--stage');p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    if a.action=='early_evaluate':early_evaluate();return
    if a.action=='worker':worker(a.stage,a.rank);return
    if a.action=='prepare':prepare(a.stage);return
    b.collect=locked_collect;s.evaluate=locked_evaluate
    while not (m.ROOT/'pipeline_complete.json').exists():time.sleep(5)
    for stage in STAGES:
        done=m.ROOT/stage/'controller_complete.json'
        if done.exists() and c.read(done)['complete']:continue
        prepare(stage); b.MODULE=MODULE; b.controller(stage)
    c.atomic(m.ROOT/'fivek_pipeline_complete.json',dict(complete=True,stages=STAGES,selection_samples=5000,final_samples=5000,final_seed=2026091451))
    subprocess.run([sys.executable,'-u','-m','experiments.raev2_shallow_ig_20260914.report_fivek'],check=True,cwd=c.WORK,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    print('5K coefficient selection and fresh final5K complete',flush=True)

if __name__=='__main__':main()
