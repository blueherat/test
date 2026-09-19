"""Balanced 1K amendment: preserve old valid batches, record divergent coefficients."""
import argparse,concurrent.futures,fcntl,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.raev2_shallow_ig_20260914 import sample as s
c,m=s.c,s.m
MODULE='experiments.raev2_shallow_ig_20260914.balanced'
STAGES=('balanced_1000','balanced_refine_1000','confirm_5k')


def bank():
    root=m.ROOT/'balanced_tuning_inputs';root.mkdir(parents=True,exist_ok=True)
    if not (root/'complete.json').exists():
        oldx,oldy=s.bank('tune_heads');x=np.load(oldx,mmap_mode='r');y=np.load(oldy)
        rng=np.random.default_rng(2026091432);p=root/'noise.npy'
        a=np.lib.format.open_memmap(p.with_suffix('.tmp'),mode='w+',dtype=np.float32,shape=(1000,1024,16,16))
        for start in range(0,400,8):
            v=rng.standard_normal((8,1024,16,16),dtype=np.float32);assert np.array_equal(v,x[start:start+8]);a[start:start+8]=v
        labels=rng.permutation(1000).astype(np.int64);assert np.array_equal(labels[:400],y)
        for start in range(400,1000,8):a[start:start+8]=rng.standard_normal((8,1024,16,16),dtype=np.float32)
        a.flush();del a;p.with_suffix('.tmp').replace(p);np.save(root/'labels.npy',labels)
        c.atomic(root/'complete.json',dict(complete=True,seed=2026091432,samples=1000,
            algorithm='Original 400 noise then permutation1000; append600 noise from continued RNG; preserve original400 exactly',
            inputs={str(root/k):c.sha(root/k) for k in ('noise.npy','labels.npy')},
            origin={str(p):c.sha(p) for p in (oldx,oldy)}))
    record=c.read(root/'complete.json')
    for group in ('inputs','origin'):
        for p,h in record[group].items():assert c.sha(p)==h
    return root/'noise.npy',root/'labels.npy'


def rows(stages):
    result=[]
    for stage in stages:
        assert c.read(m.ROOT/stage/'controller_complete.json')['complete']
        for cfg in c.read(m.ROOT/stage/'request.json')['configs']:
            out=m.ROOT/stage/cfg['arm']
            if (out/'invalid.json').exists():continue
            result.append(dict(cfg,stage=stage,fid=c.read(out/'metrics.json')[0]['fid']))
    return result


def prepare(stage):
    root=m.ROOT/stage;root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists():return s.verify(stage)
    req=dict(c.read(m.ROOT/'tune_heads/request.json'));selection={}
    if stage=='balanced_1000':
        cfgs=[s.config('incumbent',w) for w in s.COARSE]+[s.config(k,w) for k in s.FAMILIES[1:] for w in s.COARSE if w>1]
    elif stage=='balanced_refine_1000':
        available=rows(('balanced_1000',));cfgs=[];detail={}
        for kind in s.FAMILIES:
            best=min([r for r in available if r['kind']==kind or r['w']==1],key=lambda r:(r['fid'],r['w']))
            w=best['w'];i=s.COARSE.index(w);values=[]
            if i:values.append((w+s.COARSE[i-1])/2)
            values.append((w+s.COARSE[i+1])/2 if i+1<len(s.COARSE) else 4.)
            cfgs.extend(s.config(kind,v) for v in values);detail[kind]=dict(coarse_best=best,candidates=values)
        path=m.ROOT/'balanced_refinement_selection.json';c.atomic(path,detail);selection[str(path)]=c.sha(path)
    else:
        available=rows(STAGES[:2]);detail={}
        for kind in s.FAMILIES:
            best=min([r for r in available if r['kind']==kind or r['w']==1],key=lambda r:(r['fid'],r['w']))
            detail[kind]=dict(best_w=best['w'],tuning_fid=best['fid'],selection_arm=best['arm'],selection_stage=best['stage'])
        path=m.ROOT/'selected_coefficients.json';c.atomic(path,detail);selection[str(path)]=c.sha(path)
        cfgs=[s.config(k,v['best_w']) for k,v in detail.items()]
    paths=s.bank('confirm_5k') if stage=='confirm_5k' else bank()
    req.update(stage=stage,samples=5000 if stage=='confirm_5k' else 1000,configs=cfgs,noise=str(paths[0]),labels=str(paths[1]),
        inputs={str(p):c.sha(p) for p in paths},selection=selection,
        selection_rule='Balanced1000 paired search and local refinement, w1 allowed; nonfinite arms invalid; locked independent5000')
    req['sources']=dict(req['sources'])
    for p in (Path(__file__).resolve(),c.WORK/'docs/RAEV2_SHALLOW_IG_BALANCED_SEARCH_20260914_ZH.md'):req['sources'][str(p)]=c.sha(p)
    req['prefix_records']={}
    if stage=='balanced_1000':
        for parent_stage in ('tune_native','tune_heads'):s.verify(parent_stage)
        for cfg in cfgs:
            parent=m.ROOT/('tune_native' if cfg['kind']=='incumbent' else 'tune_heads')
            rh=c.sha(parent/'request.json');records={}
            for p in sorted((parent/cfg['arm']).glob('batch*.npz')):
                with np.load(p) as d:
                    start=int(d['start']);assert str(d['request_sha256'])==rh
                    assert start<400 and len(d['pixels'])==16 and np.isfinite(d['latents']).all()
                records[str(start)]=dict(path=str(p),sha256=c.sha(p),source_request_sha256=rh)
            req['prefix_records'][cfg['arm']]=records
        for p in (m.ROOT/'tune_native/request.json',m.ROOT/'tune_heads/request.json'):
            req['selection'][str(p)]=c.sha(p)
    c.atomic(root/'request.json',req);print('Prepared',stage,len(cfgs),flush=True);return req


def record_path(req,cfg,start):
    origin=req.get('prefix_records',{}).get(cfg['arm'],{}).get(str(start))
    return Path(origin['path']) if origin else m.ROOT/req['stage']/cfg['arm']/f'batch{start:06d}.npz'


@torch.inference_mode()
def worker(stage,rank):
    req=s.verify(stage);rt=c.runtime('raev2');heads=s.load_heads(rt)
    captures={d:m.Capture(rt,d) for d in (4,8)};calls=m.Counts(rt,heads)
    noise=np.load(req['noise'],mmap_mode='r');labels=np.load(req['labels']);rh=c.sha(m.ROOT/stage/'request.json');warmed=set()
    while (job:=s.pop(stage)) is not None:
        cfg=job['config'];out=m.ROOT/stage/cfg['arm'];out.mkdir(parents=True,exist_ok=True)
        if (out/'invalid.json').exists():continue
        start=job['start']
        try:
            if cfg['arm'] not in warmed:
                s.trajectory(rt,heads,captures,c.cuda(noise[:2]),c.cuda(labels[:2]),cfg);warmed.add(cfg['arm'])
            for start in range(job['start'],job['stop'],req['batch']):
                if (out/'invalid.json').exists():break
                if record_path(req,cfg,start).exists():continue
                stop=min(start+req['batch'],req['samples']);x,y=c.cuda(noise[start:stop]),c.cuda(labels[start:stop])
                calls.reset();torch.cuda.synchronize();begin=time.perf_counter()
                z,n=s.trajectory(rt,heads,captures,x,y,cfg);pixels=rt.decode(z);torch.cuda.synchronize();seconds=time.perf_counter()-begin
                expected=99 if cfg['kind']!='incumbent' and cfg['w']!=1 else 0
                assert n==dict(full=100,prefix=0) and calls.blocks==[100]*30 and sum(calls.heads.values())==expected
                s.save_npz(out/f'batch{start:06d}.npz',pixels=pixels,latents=z.cpu().numpy(),labels=labels[start:stop],start=start,
                    seconds=seconds,full=100,head_calls=expected,block_calls=np.array(calls.blocks),request_sha256=rh,noise_sha256=c.array_sha(noise[start:stop]))
                c.atomic(m.ROOT/stage/f'progress{rank}.json',dict(arm=cfg['arm'],start=start,stop=stop,phase='sampling'))
        except FloatingPointError as error:
            if stage=='confirm_5k':raise
            c.atomic(out/'invalid.json',dict(invalid=True,config=cfg,start=start,error=str(error),request_sha256=rh,
                reason='Nonfinite trajectory invalidates coefficient; no clipping, sample replacement or partial FID'))
            print('Invalid coefficient',cfg,start,str(error),flush=True)
    for cap in captures.values():cap.close()
    calls.close()


def collect(stage,cfg,req):
    out=m.ROOT/stage/cfg['arm'];rh=c.sha(m.ROOT/stage/'request.json')
    if (out/'invalid.json').exists() or (out/'summary.json').exists():return
    noise=np.load(req['noise'],mmap_mode='r');labels=np.load(req['labels']);images=[];records=[];seconds=0.;reused=0;coverage=[]
    for start in range(0,req['samples'],req['batch']):
        path=record_path(req,cfg,start);origin=req.get('prefix_records',{}).get(cfg['arm'],{}).get(str(start))
        digest=c.sha(path)
        if origin:assert digest==origin['sha256']
        source_rh=origin['source_request_sha256'] if origin else rh
        with np.load(path) as d:
            stop=start+len(d['pixels']);coverage.extend(range(start,stop))
            assert int(d['start'])==start and str(d['request_sha256'])==source_rh
            assert str(d['noise_sha256'])==c.array_sha(noise[start:stop]) and np.array_equal(d['labels'],labels[start:stop])
            assert np.isfinite(d['latents']).all();images.append(d['pixels']);seconds+=float(d['seconds'])
            if origin:reused+=stop-start
        records.append(dict(path=str(path),sha256=digest,source_request_sha256=source_rh,reused=bool(origin)))
    assert coverage==list(range(req['samples']));s.save_npz(out/'samples.npz',arr_0=np.concatenate(images))
    c.atomic(out/'summary.json',dict(complete=True,samples=req['samples'],reused_prefix_samples=reused,new_samples=req['samples']-reused,
        seconds=seconds,full_calls_per_output=100,prefix_calls_per_output=0,records=records,request_sha256=rh,samples_sha256=c.sha(out/'samples.npz')))


def controller(stage):
    req=s.verify(stage);root=m.ROOT/stage
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);jobs=[]
        for cfg in req['configs']:
            if (root/cfg['arm']/'invalid.json').exists():continue
            for start in range(0,req['samples'],128):
                stop=min(start+128,req['samples'])
                if any(not record_path(req,cfg,k).exists() for k in range(start,stop,16)):jobs.append(dict(config=cfg,start=start,stop=stop))
        c.atomic(root/'queue.json',jobs);procs=[];streams=[]
        for rank in range(4):
            f=(root/f'worker{rank}.log').open('a');streams.append(f)
            procs.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--stage',stage,'--rank',str(rank)],cwd=c.WORK,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')))
        while any(p.poll() is None for p in procs):
            codes=[p.poll() for p in procs]
            if any(x not in (None,0) for x in codes):
                for p in procs:
                    if p.poll() is None:p.terminate()
                break
            c.atomic(root/'status.json',dict(complete=False,phase='sampling',remaining_shards=len(c.read(root/'queue.json'))));time.sleep(5)
        codes=[p.wait() for p in procs]
        for f in streams:f.close()
        assert not any(codes),codes
        valid=[];invalid=[]
        for cfg in req['configs']:
            collect(stage,cfg,req)
            (invalid if (root/cfg['arm']/'invalid.json').exists() else valid).append(cfg)
        def task(gpu):
            for cfg in valid[gpu::4]:s.evaluate(stage,cfg,gpu)
        c.atomic(root/'status.json',dict(complete=False,phase='evaluation',valid=len(valid),invalid=len(invalid)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(task,range(4)))
        c.atomic(root/'controller_complete.json',dict(complete=True,arms=len(valid),invalid=invalid))
        c.atomic(root/'status.json',dict(complete=True,phase='complete',valid=len(valid),invalid=len(invalid)))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pipeline','worker']);p.add_argument('--stage');p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    if a.action=='worker':worker(a.stage,a.rank);return
    c.atomic(m.ROOT/'balanced_handoff_ready.json',dict(complete=True,reason='Prior400 controller exited on nonfinite high-w candidate; preserve all valid prefix batches'))
    for stage in STAGES:
        done=m.ROOT/stage/'controller_complete.json'
        if done.exists() and c.read(done)['complete']:continue
        prepare(stage);controller(stage)
    c.atomic(m.ROOT/'pipeline_complete.json',dict(complete=True,stages=STAGES,selection_samples=1000,final_samples_per_family=5000))
    print('Balanced search and independent5K complete',flush=True)

if __name__=='__main__':main()
