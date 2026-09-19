"""Sharded, paired coefficient search followed by held-out 5K evaluation."""
import argparse,fcntl,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from experiments.raev2_shallow_ig_20260914 import core as m
from experiments.raev2_shallow_ig_20260914 import continue_depth8 as d8
from experiments.weak_reference_20260914 import cross_run as evaluate_base
from experiments.weak_reference_20260914.cross_run import save_npz
c=m.c
MODULE='experiments.raev2_shallow_ig_20260914.sample'
FAMILIES=('incumbent','native4','mlp4','mlp8')
COARSE=(1.,1.2,1.4,1.6,1.78,2.,2.4,3.)
REFERENCE=evaluate_base.REFERENCE
LEGACY=c.EXPS/'raev2_context_5k_20260914/raev2/confirm_5000'


def config(kind,w):return dict(arm=f'{kind}_w{w:.4f}'.replace('.','p'),kind=kind,w=float(w))


def configs(stage):
    if stage=='tune_native':return [config('incumbent',w) for w in COARSE]
    if stage=='tune_heads':return [config(k,w) for k in FAMILIES[1:] for w in COARSE if w>1]
    if stage=='tune_refine':
        rows=all_tune_rows(stages=('tune_native','tune_heads'));result=[];selection={}
        for kind in FAMILIES:
            trials=[r for r in rows if r['kind']==kind or r['w']==1]
            best=min(trials,key=lambda r:(r['fid'],r['w']));w=best['w'];j=COARSE.index(w)
            candidates=[]
            if j>0:candidates.append((w+COARSE[j-1])/2)
            if j+1<len(COARSE):candidates.append((w+COARSE[j+1])/2)
            else:candidates.append(4.)
            selection[kind]=dict(coarse_best=best,candidates=candidates)
            result.extend(config(kind,x) for x in candidates)
        c.atomic(m.ROOT/'refinement_selection.json',selection)
        return result
    if stage=='confirm_5k':
        rows=all_tune_rows();selected={}
        for kind in FAMILIES:
            trials=[r for r in rows if r['kind']==kind or r['w']==1]
            best=min(trials,key=lambda r:(r['fid'],r['w']))
            selected[kind]=dict(best_w=best['w'],tuning_fid=best['fid'],selection_arm=best['arm'],selection_stage=best['stage'])
        c.atomic(m.ROOT/'selected_coefficients.json',selected)
        return [config(k,v['best_w']) for k,v in selected.items()]
    raise ValueError(stage)


def all_tune_rows(stages=('tune_native','tune_heads','tune_refine')):
    rows=[]
    for stage in stages:
        root=m.ROOT/stage;request=c.read(root/'request.json')
        assert c.read(root/'controller_complete.json')['complete']
        for cfg in request['configs']:
            metric=c.read(root/cfg['arm']/'metrics.json')[0]
            rows.append(dict(cfg,stage=stage,fid=metric['fid']))
    return rows


def bank(stage):
    if stage=='confirm_5k':
        # Independent of coefficient search; retained bank permits exact legacy comparisons.
        return LEGACY/'inputs/first.npy',LEGACY/'inputs/labels.npy'
    root=m.ROOT/'tuning_inputs';root.mkdir(parents=True,exist_ok=True)
    if not (root/'complete.json').exists():
        rng=np.random.default_rng(2026091432)
        p=root/'noise.npy';a=np.lib.format.open_memmap(p.with_suffix('.tmp'),mode='w+',dtype=np.float32,shape=(400,1024,16,16))
        for start in range(0,400,8):a[start:start+8]=rng.standard_normal((8,1024,16,16),dtype=np.float32)
        a.flush();del a;p.with_suffix('.tmp').replace(p)
        np.save(root/'labels.npy',rng.permutation(1000)[:400].astype(np.int64))
        c.atomic(root/'complete.json',dict(complete=True,samples=400,seed=2026091432,
            inputs={str(root/p):c.sha(root/p) for p in ('noise.npy','labels.npy')}))
    record=c.read(root/'complete.json')
    for p,digest in record['inputs'].items():assert c.sha(p)==digest
    return root/'noise.npy',root/'labels.npy'


def prepare(stage):
    root=m.ROOT/stage;root.mkdir(parents=True,exist_ok=True)
    choices=configs(stage);paths=bank(stage);need_heads=any(x['kind']!='incumbent' for x in choices)
    sources=c.source_manifest([Path(__file__),Path(m.__file__),Path(m.old.__file__),Path(d8.__file__),
        c.WORK/'experiments/raev2_shallow_ig_20260914/train.py',
        c.WORK/'experiments/evaluate_raev2_official_samples.py',
        c.WORK/'docs/RAEV2_SHALLOW_IG_50K_SEARCH_20260914_ZH.md'])
    for p in evaluate_base.sources('raev2'):sources[str(p)]=c.sha(p)
    assets={str(p):c.sha(p) for p in c.asset_paths('raev2')}
    for p in (REFERENCE,Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'),
            Path('/home/zhoushunyu/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth')):assets[str(p)]=c.sha(p)
    heads={}
    if need_heads:
        for folder in (m.TRAIN,d8.TRAIN):
            assert c.read(folder/'summary.json')['complete'] and c.read(folder/'summary.json')['steps']==50000
            for p in ('head.pt','summary.json','request.json'):heads[str(folder/p)]=c.sha(folder/p)
    check=m.ROOT/('heads_preflight.json' if need_heads else 'native_preflight.json')
    assert c.read(check)['passed']
    req=dict(stage=stage,samples=5000 if stage=='confirm_5k' else 400,batch=16,configs=choices,
        noise=str(paths[0]),labels=str(paths[1]),inputs={str(p):c.sha(p) for p in paths},
        sources=sources,assets=assets,heads=heads,check={str(check):c.sha(check)},reference=str(REFERENCE),
        search_variable='extrapolation w in weak+w*(strong-weak); implemented full+(w-1)*(full-weak)',
        sampler='Euler100 shift8, window [.1,1], FP32 velocity, BF16 model, TF32',
        evaluation='Nanogen FP32 Inception, batch64, same legacy reference; all samples included',
        selection_rule='400 paired samples for search only; independent 5K bank for locked selected settings')
    if stage=='tune_refine':req['selection']={str(m.ROOT/'refinement_selection.json'):c.sha(m.ROOT/'refinement_selection.json')}
    if stage=='confirm_5k':req['selection']={str(m.ROOT/'selected_coefficients.json'):c.sha(m.ROOT/'selected_coefficients.json')}
    path=root/'request.json'
    if path.exists():assert c.read(path)==req
    else:c.atomic(path,req)
    print('Prepared',stage,len(choices),'arms',req['samples'],flush=True)


def verify(stage):
    req=c.read(m.ROOT/stage/'request.json')
    for group in ('sources','assets','heads','inputs','check','selection'):
        for p,digest in req.get(group,{}).items():assert c.sha(p)==digest,(group,p)
    return req


def load_heads(rt):
    a=m.load(rt);b=m.old.make_head(rt).eval().requires_grad_(False)
    state=torch.load(d8.TRAIN/'head.pt',map_location='cpu',weights_only=True)
    assert state['step']==50000 and state['request_sha256']==c.sha(d8.TRAIN/'request.json')
    assert c.sha(d8.TRAIN/'head.pt')==c.read(d8.TRAIN/'summary.json')['head_sha256']
    b.load_state_dict(state['ema']['context']);return dict(native4=a['native'],mlp4=a['mlp'],mlp8=b)


def trajectory(rt,heads,captures,x,y,cfg):
    kind=cfg['kind'];a=cfg['w']-1
    if kind=='incumbent':return m.sample(rt,{},None,x,y,'incumbent',a)
    key='native' if kind=='native4' else 'mlp'
    return m.sample(rt,{key:heads[kind]},captures[8 if kind=='mlp8' else 4],x,y,key,a)


@torch.inference_mode()
def preflight(native_only):
    rt=c.runtime('raev2');heads={} if native_only else load_heads(rt)
    result=m.check(rt,m.make_heads(rt));captures={d:m.Capture(rt,d) for d in (4,8)}
    calls=m.Counts(rt,heads)
    x=np.load(LEGACY/'inputs/first.npy',mmap_mode='r');y=np.load(LEGACY/'inputs/labels.npy')
    x,y=c.cuda(x[:16]),c.cuda(y[:16]);calls.reset()
    z,n=trajectory(rt,heads,captures,x,y,config('incumbent',1.78));pixels=rt.decode(z)
    with np.load(LEGACY/'native_base/rank0/batch0000.npz') as d:
        assert torch.equal(z.cpu(),torch.from_numpy(d['latents']))
        assert np.array_equal(pixels,d['arr_0'])
    assert n==dict(full=100,prefix=0) and calls.blocks==[100]*30
    zero,_=trajectory(rt,heads,captures,x[:2],y[:2],config('incumbent',1))
    for kind in heads:
        calls.reset();other,n=trajectory(rt,heads,captures,x[:2],y[:2],config(kind,1))
        assert torch.equal(zero,other) and not any(calls.heads.values())
        calls.reset();other,n=trajectory(rt,heads,captures,x[:2],y[:2],config(kind,1.4))
        assert n==dict(full=100,prefix=0) and calls.blocks==[100]*30
        assert calls.heads[kind]==99 and sum(calls.heads.values())==99
        assert torch.isfinite(other).all() and not torch.equal(other,zero)
    for capture in captures.values():capture.close()
    calls.close();result.update(passed=True,legacy_native_16_latents_and_pixels_bitwise=True,
        zero_w1_exact=True,full_calls=100,extra_prefix=0,trained_heads_checked=list(heads))
    c.atomic(m.ROOT/('native_preflight.json' if native_only else 'heads_preflight.json'),result)
    print('Preflight passed',result,flush=True)


def pop(stage):
    root=m.ROOT/stage
    with (root/'queue.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);q=c.read(root/'queue.json')
        if not q:return None
        job=q.pop(0);c.atomic(root/'queue.json',q);return job


@torch.inference_mode()
def worker(stage,rank):
    req=verify(stage);rt=c.runtime('raev2')
    heads=load_heads(rt) if req['heads'] else {};captures={d:m.Capture(rt,d) for d in (4,8)};calls=m.Counts(rt,heads)
    noise=np.load(req['noise'],mmap_mode='r');labels=np.load(req['labels']);rh=c.sha(m.ROOT/stage/'request.json')
    warmed=set()
    while (job:=pop(stage)) is not None:
        cfg=job['config'];out=m.ROOT/stage/cfg['arm'];out.mkdir(parents=True,exist_ok=True)
        if cfg['arm'] not in warmed:
            trajectory(rt,heads,captures,c.cuda(noise[:2]),c.cuda(labels[:2]),cfg);warmed.add(cfg['arm'])
        for start in range(job['start'],job['stop'],req['batch']):
            path=out/f'batch{start:06d}.npz'
            if path.exists():continue
            stop=min(start+req['batch'],req['samples']);x,y=c.cuda(noise[start:stop]),c.cuda(labels[start:stop])
            calls.reset();torch.cuda.synchronize();begin=time.perf_counter()
            z,n=trajectory(rt,heads,captures,x,y,cfg);pixels=rt.decode(z);torch.cuda.synchronize();seconds=time.perf_counter()-begin
            expected=99 if cfg['kind']!='incumbent' and cfg['w']!=1 else 0
            assert n==dict(full=100,prefix=0) and calls.blocks==[100]*30 and sum(calls.heads.values())==expected
            save_npz(path,pixels=pixels,latents=z.cpu().numpy(),labels=labels[start:stop],start=start,
                seconds=seconds,full=100,head_calls=expected,block_calls=np.array(calls.blocks),
                request_sha256=rh,noise_sha256=c.array_sha(noise[start:stop]))
            c.atomic(m.ROOT/stage/f'progress{rank}.json',dict(arm=cfg['arm'],start=start,stop=stop,phase='sampling'))
    for cap in captures.values():cap.close()
    calls.close();c.atomic(m.ROOT/stage/f'worker{rank}_complete.json',dict(complete=True))


def collect(stage,cfg,req):
    out=m.ROOT/stage/cfg['arm']
    if (out/'summary.json').exists():return
    noise=np.load(req['noise'],mmap_mode='r');labels=np.load(req['labels']);rh=c.sha(m.ROOT/stage/'request.json')
    images=[];records=[];seconds=0.;coverage=[]
    for start in range(0,req['samples'],req['batch']):
        path=out/f'batch{start:06d}.npz'
        with np.load(path) as d:
            stop=start+len(d['pixels']);assert int(d['start'])==start;coverage.extend(range(start,stop))
            assert np.array_equal(d['labels'],labels[start:stop]) and str(d['request_sha256'])==rh
            assert str(d['noise_sha256'])==c.array_sha(noise[start:stop])
            assert np.isfinite(d['latents']).all() and np.array_equal(d['block_calls'],np.full(30,100))
            images.append(d['pixels']);seconds+=float(d['seconds'])
        records.append(dict(path=str(path),sha256=c.sha(path)))
    assert coverage==list(range(req['samples']));save_npz(out/'samples.npz',arr_0=np.concatenate(images))
    c.atomic(out/'summary.json',dict(complete=True,samples=req['samples'],seconds=seconds,full_calls_per_output=100,
        prefix_calls_per_output=0,records=records,request_sha256=rh,samples_sha256=c.sha(out/'samples.npz')))


def evaluate(stage,cfg,gpu):
    out=m.ROOT/stage/cfg['arm']
    if (out/'metrics.json').exists():return
    command=[sys.executable,str(c.WORK/'experiments/evaluate_raev2_official_samples.py'),
        '--branch',cfg['arm']+'='+str(out/'samples.npz'),'--output',str(out/'metrics.csv'),'--device','cuda',
        '--batch-size','64','--fid-reference',str(REFERENCE),'--feature-cache-dir',str(out/'features')]
    with (out/'evaluation.log').open('w') as f:
        subprocess.run(command,check=True,cwd=c.WORK,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    print(stage,cfg['arm'],c.read(out/'metrics.json')[0]['fid'],flush=True)


def controller(stage,gpus):
    req=verify(stage);root=m.ROOT/stage
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);jobs=[]
        for cfg in req['configs']:
            out=root/cfg['arm']
            for start in range(0,req['samples'],128):
                stop=min(start+128,req['samples'])
                if any(not (out/f'batch{k:06d}.npz').exists() for k in range(start,stop,16)):
                    jobs.append(dict(config=cfg,start=start,stop=stop))
        c.atomic(root/'queue.json',jobs);processes=[];streams=[]
        for rank,gpu in enumerate(gpus.split(',')):
            f=(root/f'worker{rank}.log').open('a');streams.append(f)
            processes.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--stage',stage,'--rank',str(rank)],
                cwd=c.WORK,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')))
        while any(p.poll() is None for p in processes):
            codes=[p.poll() for p in processes]
            if any(x not in (None,0) for x in codes):
                for p in processes:
                    if p.poll() is None:p.terminate()
                for p in processes:p.wait()
                break
            c.atomic(root/'status.json',dict(phase='sampling',complete=False,remaining_shards=len(c.read(root/'queue.json'))))
            time.sleep(5)
        codes=[p.wait() for p in processes]
        for f in streams:f.close()
        if any(codes):
            c.atomic(root/'controller_complete.json',dict(complete=False,exit_codes=codes));raise RuntimeError(codes)
        for cfg in req['configs']:collect(stage,cfg,req)
        # Released sampling GPUs; evaluate several completed arms concurrently.
        pending=list(req['configs']);running=[];slots=gpus.split(',')
        while pending or running:
            for gpu in slots:
                if pending and not any(item[0]==gpu for item in running):
                    cfg=pending.pop(0)
                    p=subprocess.Popen([sys.executable,'-u','-m',MODULE,'evaluate','--stage',stage,'--arm',cfg['arm'],'--gpus',gpu],cwd=c.WORK)
                    running.append((gpu,p,cfg))
            for item in list(running):
                if item[1].poll() is not None:
                    assert item[1].returncode==0,item[2];running.remove(item)
            time.sleep(2)
        c.atomic(root/'controller_complete.json',dict(complete=True,exit_codes=codes,arms=len(req['configs'])))
        c.atomic(root/'status.json',dict(complete=True,phase='complete',arms=len(req['configs'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','prepare','controller','worker','evaluate'])
    p.add_argument('--stage',default='tune_native');p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--rank',type=int,default=0)
    p.add_argument('--native-only',action='store_true');p.add_argument('--arm');a=p.parse_args()
    if a.action=='preflight':preflight(a.native_only)
    elif a.action=='prepare':prepare(a.stage)
    elif a.action=='controller':controller(a.stage,a.gpus)
    elif a.action=='worker':worker(a.stage,a.rank)
    else:evaluate(a.stage,next(x for x in c.read(m.ROOT/a.stage/'request.json')['configs'] if x['arm']==a.arm),a.gpus)

if __name__=='__main__':main()
