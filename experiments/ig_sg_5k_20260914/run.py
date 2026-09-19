"""Paired SG weight search and sharded 5K with the frozen original IG models."""
import argparse,concurrent.futures,fcntl,gc,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.ig_sg_20260914 import core as m
from experiments.ig_sg_20260914 import run as old
from experiments.weak_reference_20260914.cross_run import save_npz
c=m.common
ROOT=c.EXPS/'ig_sg_5k_20260914'
MODULE='experiments.ig_sg_5k_20260914.run'
MODELS=('sit_small','jit','raev2')
OMEGAS=(.25,.5,1.,2.)


def sg_config(name,kind,omega):
    cfg=dict(m.configs(name)[1 if kind=='sg' else 2]);w=1+omega
    cfg.update(arm=f'ig_{kind}_w{w:.4f}'.replace('.','p'),omega=float(omega),w=float(w))
    return cfg


def choices(phase,name):
    if phase=='tune':return [m.configs(name)[0]]+[sg_config(name,k,w) for k in ('sg','log') for w in OMEGAS]
    previous=c.read(ROOT/'tune'/name/'request.json');assert previous['samples']==5000;rows=[]
    for cfg in previous['configs']:
        if (ROOT/'tune'/name/cfg['arm']/'invalid.json').exists():continue
        metric=c.read(ROOT/'tune'/name/cfg['arm']/'metrics.json')[0]
        rows.append(dict(cfg,fid=metric['fid'],source_phase='tune'))
    if phase=='refine':
        result=[];selection={}
        for kind in ('sg','log'):
            best=min([r for r in rows if r['kind']==kind],key=lambda r:(r['fid'],r['omega']))
            w=best['omega'];i=OMEGAS.index(w)
            candidates=[w/2 if i==0 else (w+OMEGAS[i-1])/2,
                        3. if i==len(OMEGAS)-1 else (w+OMEGAS[i+1])/2]
            result.extend(sg_config(name,kind,v) for v in candidates)
            selection[kind]=dict(coarse_best=best,refine_omegas=candidates)
        c.atomic(ROOT/'selection'/f'{name}_refine.json',selection);return result
    assert phase=='confirm_5k'
    refined=c.read(ROOT/'refine'/name/'request.json');assert refined['samples']==5000
    for cfg in refined['configs']:
        if (ROOT/'refine'/name/cfg['arm']/'invalid.json').exists():continue
        metric=c.read(ROOT/'refine'/name/cfg['arm']/'metrics.json')[0]
        rows.append(dict(cfg,fid=metric['fid'],source_phase='refine'))
    selected={k:min([r for r in rows if r['kind']==k],key=lambda r:(r['fid'],r['omega'])) for k in ('sg','log')}
    c.atomic(ROOT/'selection'/f'{name}_final.json',selected)
    base=m.configs(name)
    return [base[0],sg_config(name,'sg',selected['sg']['omega']),sg_config(name,'log',selected['log']['omega']),base[3],base[4]]


def input_bank(phase,name):
    root=ROOT/'inputs'/('confirm_5k' if phase in ('tune','refine') else 'final_v2')/name
    root.mkdir(parents=True,exist_ok=True);stamp=root/'complete.json'
    if not stamp.exists():
        seed=2026091441 if phase in ('tune','refine') else 2026091452
        rng=np.random.default_rng(seed);p=root/'noise.npy'
        a=np.lib.format.open_memmap(p.with_suffix('.tmp'),mode='w+',dtype=np.float32,shape=(5000,*m.SHAPES[name]))
        for start in range(0,5000,8):a[start:start+8]=rng.standard_normal((8,*m.SHAPES[name]),dtype=np.float32)
        a.flush();del a;p.with_suffix('.tmp').replace(p)
        labels=np.arange(5000,dtype=np.int64)%(100 if name=='sit_small' else 1000);rng.shuffle(labels);np.save(root/'labels.npy',labels)
        c.atomic(stamp,dict(complete=True,seed=seed,samples=5000,
            inputs={str(root/f'{k}.npy'):c.sha(root/f'{k}.npy') for k in ('noise','labels')},origin={}))
    record=c.read(stamp)
    for group in ('inputs','origin'):
        for path,h in record[group].items():assert c.sha(path)==h
    labels=np.load(root/'labels.npy');noise=np.load(root/'noise.npy',mmap_mode='r');classes=100 if name=='sit_small' else 1000
    assert record['samples']==5000 and noise.shape==(5000,*m.SHAPES[name]) and noise.dtype==np.float32
    assert labels.shape==(5000,) and np.array_equal(np.bincount(labels,minlength=classes),np.full(classes,5000//classes))
    return root,record


def prepare(phase):
    for name in MODELS:
        cfgs=choices(phase,name);bank,record=input_bank(phase,name);root=ROOT/phase/name;root.mkdir(parents=True,exist_ok=True)
        source_root=m.ROOT/'screen_1k'/name;prior=old.verify(source_root)
        sources=dict(prior['sources'])
        for p in (Path(__file__),c.WORK/'docs/IG_SG_5K_SEARCH_PROTOCOL_20260914_ZH.md'):sources[str(p.resolve())]=c.sha(p)
        reuse={}  # Earlier1K outputs are diagnostic only; all current coefficients generate5000.
        req=dict(model=name,phase=phase,samples=record['samples'],seed=record['seed'],batch=16 if name=='sit_small' else 32,
            configs=cfgs,sources=sources,assets=prior['assets'],inputs=record['inputs'],input_origin=record['origin'],
            noise=str(bank/'noise.npy'),labels=str(bank/'labels.npy'),reused_arms=reuse,
            reference=prior['reference'],ig_schedule=prior['ig_schedule'],precision=prior['precision'],
            definition='v_out=v_IG+(w_SG-1)*(v_strong-v_reference); omega=w_SG-1. Raw strong residual, original frozen IG coefficients.',
            selection_rule='Every coarse and refined coefficient uses5000 paired samples; fresh seed2026091452 final5K after lock. IG-only separate.')
        if phase=='refine':req['selection']={str(ROOT/'selection'/f'{name}_refine.json'):c.sha(ROOT/'selection'/f'{name}_refine.json')}
        if phase=='confirm_5k':req['selection']={str(ROOT/'selection'/f'{name}_final.json'):c.sha(ROOT/'selection'/f'{name}_final.json')}
        if phase!='tune':
            req.setdefault('selection',{})
            for prior_phase in (('tune',) if phase=='refine' else ('tune','refine')):
                parent=ROOT/prior_phase/name
                for asset in [parent/'request.json',*sorted(parent.glob('*/metrics.json')),*sorted(parent.glob('*/invalid.json'))]:
                    req['selection'][str(asset)]=c.sha(asset)
        path=root/'request.json'
        if path.exists():assert c.read(path)==req
        else:c.atomic(path,req)
        print('Prepared',phase,name,len(cfgs),'arms',record['samples'],flush=True)


def verify(phase,name):
    req=c.read(ROOT/phase/name/'request.json')
    for group in ('sources','assets','inputs','input_origin','selection'):
        for p,h in req.get(group,{}).items():assert c.sha(p)==h,(group,p)
    for item in req['reused_arms'].values():
        source=Path(item['out'])
        assert c.sha(source/'summary.json')==item['summary_sha256']
        assert c.sha(source.parent/'request.json')==item['original_request_sha256']
        assert c.sha(source/'samples.npz')==item['samples_sha256'] and c.sha(source/'metrics.json')==item['metrics_sha256']
        for p,h in item['feature_files'].items():assert c.sha(p)==h
    return req


def pop(phase):
    root=ROOT/phase
    with (root/'queue.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);jobs=c.read(root/'queue.json')
        if not jobs:return None
        job=jobs.pop(0);c.atomic(root/'queue.json',jobs);return job


@torch.inference_mode()
def worker(phase,rank):
    rt=None;verified={};warmed=set()
    while (job:=pop(phase)) is not None:
        name,cfg=job['model'],job['config']
        if name not in verified:verified[name]=verify(phase,name)
        req=verified[name];root=ROOT/phase/name;out=root/cfg['arm'];out.mkdir(parents=True,exist_ok=True)
        if (out/'invalid.json').exists():continue
        if rt is None or rt.name!=name:
            if rt is not None:rt.close()
            del rt;gc.collect();torch.cuda.empty_cache();rt=m.Runtime(name)
        noise=np.load(req['noise'],mmap_mode='r');labels=np.load(req['labels']);rh=c.sha(root/'request.json')
        start=job['start']
        try:
            if (name,cfg['arm']) not in warmed:
                m.sample(rt,torch.from_numpy(noise[:2].copy()).cuda(),torch.from_numpy(labels[:2]).cuda(),cfg);warmed.add((name,cfg['arm']))
            for start in range(job['start'],job['stop'],req['batch']):
                if (out/'invalid.json').exists():break
                path=out/f'batch{start:06d}.npz'
                if path.exists():continue
                stop=min(start+req['batch'],req['samples']);x=torch.from_numpy(noise[start:stop].copy()).cuda();y=torch.from_numpy(labels[start:stop]).cuda()
                torch.cuda.synchronize();begin=time.perf_counter();z,counts=m.sample(rt,x,y,cfg,trace=start==0);pixels=rt.decode(z)
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                save_npz(path,pixels=pixels,latents=z.cpu().numpy(),labels=labels[start:stop],start=start,full=counts['full'],
                    prefix=counts['prefix'],trace=counts['trace'],seconds=seconds,request_sha256=rh,noise_sha256=c.array_sha(noise[start:stop]))
                c.atomic(ROOT/phase/f'progress{rank}.json',dict(model=name,arm=cfg['arm'],start=start,stop=stop,total=req['samples'],phase='sampling'))
        except FloatingPointError as error:
            if phase=='confirm_5k':raise
            failure=dict(invalid=True,config=cfg,start=start,error=str(error),request_sha256=rh,
                reason='Trajectory failed finite/magnitude check; whole coefficient invalid, no partial FID')
            with (out/'failure.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                c.atomic(out/f'failure_rank{rank}_start{start:06d}.json',failure)
                if not (out/'invalid.json').exists():c.atomic(out/'invalid.json',failure)
            print('Invalid coefficient',name,cfg,start,flush=True)
    if rt is not None:rt.close()
    c.atomic(ROOT/phase/f'worker{rank}_complete.json',dict(complete=True))


def collect(phase,name,cfg,req):
    out=ROOT/phase/name/cfg['arm']
    if (out/'summary.json').exists():return
    images=[];records=[];full=[];seconds=0.;noise=np.load(req['noise'],mmap_mode='r');labels=np.load(req['labels'])
    rh=c.sha(ROOT/phase/name/'request.json')
    for start in range(0,req['samples'],req['batch']):
        path=out/f'batch{start:06d}.npz'
        with np.load(path) as d:
            stop=start+len(d['pixels']);assert int(d['start'])==start and str(d['request_sha256'])==rh
            assert np.array_equal(d['labels'],labels[start:stop]) and str(d['noise_sha256'])==c.array_sha(noise[start:stop])
            assert np.isfinite(d['latents']).all() and int(d['prefix'])==0
            images.append(d['pixels']);full.append(int(d['full']));seconds+=float(d['seconds'])
        records.append(dict(path=str(path),sha256=c.sha(path)))
    assert len(set(full))==1;images=np.concatenate(images);assert len(images)==req['samples'];save_npz(out/'samples.npz',arr_0=images)
    c.atomic(out/'summary.json',dict(complete=True,samples=len(images),reused=False,records=records,full_calls_per_output=full[0],prefix_calls_per_output=0,
        seconds=seconds,request_sha256=rh,samples_sha256=c.sha(out/'samples.npz')))


def evaluate(phase,name,cfg,gpu):
    out=ROOT/phase/name/cfg['arm']
    if (out/'metrics.json').exists():return
    command=[sys.executable,'-u','-m',MODULE,'evaluate_one','--phase',phase,'--model',name,'--arm',cfg['arm']]
    subprocess.run(command,check=True,cwd=c.WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))


def controller(phase):
    root=ROOT/phase;reqs={name:verify(phase,name) for name in MODELS}
    with (root/'controller.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);jobs=[]
        for name,req in reqs.items():
            for cfg in req['configs']:
                if cfg['arm'] in req['reused_arms'] or (root/name/cfg['arm']/'invalid.json').exists():continue
                for start in range(0,req['samples'],256):
                    stop=min(start+256,req['samples'])
                    if any(not (root/name/cfg['arm']/f'batch{i:06d}.npz').exists() for i in range(start,stop,req['batch'])):
                        jobs.append(dict(model=name,config=cfg,start=start,stop=stop))
        weights={'sit_small':.2,'jit':2,'raev2':8}
        jobs.sort(key=lambda j:weights[j['model']]*j['config']['steps']*{'baseline':1,'sg':2,'log':3}[j['config']['kind']],reverse=True)
        c.atomic(root/'queue.json',jobs);procs=[];streams=[]
        for rank in range(4):
            stream=(root/f'worker{rank}.log').open('a');streams.append(stream)
            procs.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--phase',phase,'--rank',str(rank)],
                cwd=c.WORK,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')))
        while any(p.poll() is None for p in procs):
            codes=[p.poll() for p in procs]
            if any(v not in (None,0) for v in codes):
                for p in procs:
                    if p.poll() is None:p.terminate()
                for p in procs:p.wait()
                break
            c.atomic(root/'status.json',dict(complete=False,phase='sampling',remaining_shards=len(c.read(root/'queue.json'))));time.sleep(5)
        codes=[p.wait() for p in procs]
        for stream in streams:stream.close()
        if any(codes):
            c.atomic(root/'controller_complete.json',dict(complete=False,exit_codes=codes));raise RuntimeError(codes)
        all_arms=[]
        for name,req in reqs.items():
            for cfg in req['configs']:
                if (root/name/cfg['arm']/'invalid.json').exists():continue
                collect(phase,name,cfg,req);all_arms.append((name,cfg))
        def task(gpu):
            for name,cfg in all_arms[gpu::4]:evaluate(phase,name,cfg,gpu)
        c.atomic(root/'status.json',dict(complete=False,phase='evaluation',arms=len(all_arms)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(task,range(4)))
        c.atomic(root/'controller_complete.json',dict(complete=True,exit_codes=codes,arms=len(all_arms)))
        c.atomic(root/'status.json',dict(complete=True,phase='complete',arms=len(all_arms)))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','controller','worker','evaluate_one']);p.add_argument('--phase',default='tune')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--model',choices=MODELS);p.add_argument('--arm');a=p.parse_args()
    if a.action=='prepare':prepare(a.phase)
    elif a.action=='controller':controller(a.phase)
    elif a.action=='worker':worker(a.phase,a.rank)
    else:
        out=ROOT/a.phase/a.model/a.arm;old.evaluate(out,a.model)
        print(a.phase,a.model,a.arm,c.read(out/'metrics.json')[0]['fid'],flush=True)

if __name__=='__main__':main()
