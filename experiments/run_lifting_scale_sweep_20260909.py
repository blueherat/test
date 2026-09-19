"""Prepare audited banks; four GPUs cooperate on each new IG/lifting point."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.lifting_scale_sweep_20260909 import (
    ALPHAS, DATA, EXPS, MODELS, ROOT, SMALL_DATA, SMALL_REF, WORK,
    Runtime, arm_name, array_sha, asset_paths, atomic, read, sha, source_paths,
)


def expected_inputs(name):
    if name == 'sit_xl':
        r = read(EXPS / 'official_sit_baseline_control_20260909/request.json')
    elif name == 'raev2':
        r = read(EXPS / 'ig_condition_carrier_20260908/quality/native_ig/summary.json')
    elif name == 'jit':
        v = read(EXPS / 'jit_transfer_20260908/full_euler100/input_hashes.json')
        r = dict(noise_sha256=v['noise'], label_sha256=v['labels'])
    else:
        r = read(SMALL_REF / 'sampling_manifest.json')
    return {key: r[key] for key in ('noise_sha256', 'label_sha256')}


def prepare_bank(name):
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    batch = 8 if name == 'sit_small' else 4
    shape = {'sit_xl': (4,32,32), 'sit_small': (4,32,32),
             'raev2': (1024,16,16), 'jit': (3,256,256)}[name]
    noise = np.lib.format.open_memmap(out / 'noise.npy', mode='w+', dtype=np.float32, shape=(1000,*shape))
    labels = np.empty(1000, dtype=np.int64)
    if name == 'sit_xl':
        with np.load(EXPS / 'official_sit_baseline_control_20260909/inputs.npz') as old:
            noise[:] = old['noise']
            labels[:] = old['labels']
    else:
        seed = {'raev2': 202609413, 'jit': 202609831, 'sit_small': 0}[name]
        gen = torch.Generator(device='cuda').manual_seed(seed)
        for start in range(0,1000,batch):
            if name == 'sit_small':
                gen.manual_seed(start // batch)
            noise[start:start+batch] = torch.randn(batch,*shape,device='cuda',generator=gen).cpu().numpy()
            labels[start:start+batch] = (
                torch.randint(0,100,(batch,),device='cuda',generator=gen).cpu().numpy()
                if name == 'sit_small' else np.arange(start,start+batch))
    noise.flush()
    np.save(out / 'labels.npy', labels)
    reference = expected_inputs(name)
    assert array_sha(noise) == reference['noise_sha256'], name
    assert array_sha(labels) == reference['label_sha256'], name
    return dict(**reference, batch=batch, samples=1000, shape=shape,
                noise_file_sha256=sha(out / 'noise.npy'), labels_file_sha256=sha(out / 'labels.npy'))


def reused_records(name, bank):
    records = {}

    def add(method, alpha, pixels, metric, files, *, seconds=None, full=None, prefix=None):
        pixels = Path(pixels)
        assert pixels.exists(), pixels
        ph = sha(pixels)
        if metric.get('sample_sha256'):
            assert metric['sample_sha256'] == ph, pixels
        rec = dict(model=name, method=method, alpha=alpha, arm=arm_name(method,alpha),
                   reused=True, complete=True, samples=1000, fid=metric['fid'], metrics=metric,
                   sample_path=str(pixels), sample_sha256=ph, source_files={str(p):sha(p) for p in files},
                   noise_sha256=bank['noise_sha256'], label_sha256=bank['label_sha256'],
                   sum_batch_gpu_seconds=seconds, full_calls_per_image=full, prefix_calls_per_image=prefix)
        records[rec['arm']] = rec

    if name == 'raev2':
        for alpha, directory in [(.5,'ig_capacity_lifting_constant_fourcard_20260908/a050_constant'),
                                 (.65,'ig_capacity_lifting_constant_fourcard_20260908/a065_constant'),
                                 (.78,'ig_capacity_lifting_fourcard_20260908/a078_constant'),
                                 (.85,'ig_capacity_lifting_constant_fourcard_20260908/a085_constant'),
                                 (.9,'ig_capacity_lifting_constant_fourcard_20260908/a090_constant')]:
            d = EXPS / directory
            r = read(d / 'result.json')
            assert r['complete'] and r['coverage_verified']
            for rank in range(4):
                s = read(d / f'rank{rank}/summary.json')
                assert s['complete'] and s['noise_sha256'] == bank['noise_sha256']
                assert s['label_sha256'] == bank['label_sha256']
            add('lifting',alpha,d/'samples.npz',r['method'],[d/'result.json',d/'fid.json'],
                seconds=r['sum_batch_gpu_seconds'],full=298,prefix=198)
        d = EXPS / 'ig_condition_carrier_20260908/quality'
        metric = next(r for r in read(d/'fid.json') if r['branch']=='native_ig')
        add('ig',.78,d/'native_ig/samples.npz',metric,[d/'fid.json',d/'native_ig/summary.json'],full=100,prefix=0)
    if name == 'jit':
        d = EXPS / 'jit_transfer_20260908/full_euler100'
        add('ig',0.,d/'samples.npz',read(d/'fid.json')[0],[d/'fid.json',d/'input_hashes.json'],full=100,prefix=0)
        for method in ('ig','lifting'):
            for path in sorted((EXPS/'jit_fine_sweep_20260909').glob(f'{method}_e*_l0.000_m1_n1000_s202609831/result.json')):
                r = read(path)
                if r.get('reused'):
                    continue
                assert r['samples']==1000 and r['noise_sha256']==bank['noise_sha256']
                assert r['label_sha256']==bank['label_sha256'] and r['late']==0
                add(method,r['early'],path.parent/'samples.npz',r['metrics'],[path,path.parent/'fid.json'],
                    seconds=r['sum_batch_gpu_seconds'],full=r['full_calls_per_image'],prefix=r['prefix_calls_per_image'])
    if name == 'sit_small':
        d = EXPS / 'small_sit_best_config_lifting_20260909/lifting_best_schedule'
        r = read(d/'result.json')
        assert r['complete'] and r['noise_sha256']==bank['noise_sha256'] and r['label_sha256']==bank['label_sha256']
        metric = r['metrics']
        add('lifting',.7,d/'samples_n1000.npz',metric,[d/'result.json',d/'fid.json'],
            seconds=r['sum_batch_gpu_seconds'],full=r['full_calls_per_image'],prefix=r['prefix_calls_per_image'])
    return records


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    sources = [Path(__file__), WORK/'docs/LIFTING_WIDE_SCALE_PROTOCOL_20260909_ZH.md',
               WORK/'experiments/evaluate_raev2_official_samples.py', WORK/'experiments/compute_adm_fid.py']
    for name in MODELS:
        sources.extend(source_paths(name))
    snapshot = ROOT/'sources'
    snapshot.mkdir()
    sources = sorted(set(sources))
    for i,p in enumerate(sources):
        (snapshot/f'{i:02d}_{p.name}').write_bytes(p.read_bytes())
    request = dict(models={}, alpha_grid=ALPHAS, samples=1000, model_order=MODELS,
                   sources={str(p):sha(p) for p in sources}, research_goal_achieved=False,
                   zero_shared=True, not_independent_confirmation=True)
    for name in MODELS:
        bank = prepare_bank(name)
        reused = reused_records(name,bank)
        points = {(method,a) for a in ALPHAS if a for method in ('ig','lifting')}
        points.add(('ig',0.))
        # Anchor points supply a same-profile ordinary counterpart to historical lifting.
        if name == 'sit_small': points.add(('ig',.7))
        if name == 'sit_xl': points.update({('ig',.35),('lifting',.35)})
        new = [dict(method=method,alpha=alpha,arm=arm_name(method,alpha))
               for method,alpha in sorted(points,key=lambda p:(p[1],p[0]))
               if arm_name(method,alpha) not in reused]
        model = dict(bank=bank,reused=reused,new_points=new,
                     assets={str(p):sha(p) for p in asset_paths(name)})
        request['models'][name]=model
        atomic(ROOT/name/'prepared.json',model)
        print(json.dumps(dict(prepared=name,new_points=len(new),reused_points=len(reused))),flush=True)
    atomic(ROOT/'request.json',request)
    atomic(ROOT/'status.json',dict(phase='prepared',research_goal_achieved=False))


def assert_pixels(rt, noise, labels, method, alpha, path, indices=None, steps=100):
    z, stats = rt.sample(noise,labels,method,alpha,steps=steps)
    pix = rt.decode(z)
    with np.load(path) as old:
        expected = old['arr_0'] if indices is None else old['arr_0'][indices]
    np.testing.assert_array_equal(pix,expected)
    return dict(method=method,alpha=alpha,steps=steps,path=str(path),pixel_sha256=array_sha(pix),counts=stats['counts'])


@torch.inference_mode()
def preflight(rt, noise, labels, rank):
    start = rank*rt.batch
    records=[]
    # Test full/prefix agreement on the same states at several native times.
    with rt.context():
        rt.labels=labels
        z=noise.double() if rt.name=='sit_xl' else noise.clone()
        for index in (0,37,80):
            t=rt.grid[index]
            full,base=rt.pair(z,t)
            torch.testing.assert_close(full.double(),rt.field(z,t,'full').double(),rtol=0,atol=0)
            torch.testing.assert_close(base.double(),rt.field(z,t,'base').double(),rtol=0,atol=0)
    zi,_=rt.sample(noise,labels,'ig',0.)
    zl,_=rt.sample(noise,labels,'lifting',0.)
    assert torch.equal(zi,zl)
    if rt.name=='sit_xl':
        p=EXPS/'official_sit_pfr_20260908/ordinary115/quality/samples.npz'
        records.append(assert_pixels(rt,noise,labels,'ig',.35,p,slice(start,start+rt.batch),steps=115))
    elif rt.name=='raev2':
        p=EXPS/f'ig_capacity_lifting_constant_fourcard_20260908/a050_constant/rank{rank}/batch{start:04d}.npz'
        records.append(assert_pixels(rt,noise,labels,'lifting',.5,p))
        p=EXPS/'ig_condition_carrier_20260908/quality/native_ig/samples.npz'
        records.append(assert_pixels(rt,noise,labels,'ig',.78,p,slice(start,start+rt.batch)))
    elif rt.name=='jit':
        for method in ('ig','lifting'):
            p=EXPS/f'jit_fine_sweep_20260909/{method}_e0.300_l0.000_m1_n1000_s202609831/rank{rank}/batch{start:04d}.npz'
            records.append(assert_pixels(rt,noise,labels,method,.3,p))
    else:
        p=EXPS/f'small_sit_best_config_lifting_20260909/lifting_best_schedule/rank{rank}/batch{start:04d}.npz'
        records.append(assert_pixels(rt,noise,labels,'lifting',.7,p))
    return dict(passed=True,rank=rank,indices=list(range(start,start+rt.batch)),
                pair_prefix_exact=True,zero_latents_exact=True,old_pixel_checks=records,
                runtime_sources=rt.sources,metadata=rt.metadata)


@torch.inference_mode()
def worker(name,rank):
    request=read(ROOT/'request.json');rh=sha(ROOT/'request.json')
    for p,h in request['sources'].items():assert sha(p)==h,p
    spec=request['models'][name];bank=spec['bank'];out=ROOT/name
    noise=np.load(out/'noise.npy',mmap_mode='r');labels=np.load(out/'labels.npy',mmap_mode='r')
    assert sha(out/'noise.npy')==bank['noise_file_sha256']
    assert sha(out/'labels.npy')==bank['labels_file_sha256']
    rt=Runtime(name)
    first=rank*rt.batch
    n=torch.from_numpy(np.array(noise[first:first+rt.batch])).cuda()
    l=torch.from_numpy(np.array(labels[first:first+rt.batch])).cuda()
    atomic(out/f'preflight_rank{rank}.json',preflight(rt,n,l,rank))
    while not (out/'preflight_passed.json').exists():time.sleep(.25)
    for point in spec['new_points']:
        d=out/point['arm']/f'rank{rank}';d.mkdir(parents=True,exist_ok=True)
        files=[]
        for start in range(first,1000,4*rt.batch):
            if (out/point['arm']/'numerical_failure.json').exists():
                break
            p=d/f'batch{start:04d}.npz'
            if p.exists():
                with np.load(p) as old:
                    assert str(old['request_sha256'])==rh
                    assert str(old['noise_sha256'])==array_sha(noise[start:start+rt.batch])
                    np.testing.assert_array_equal(old['labels'],labels[start:start+rt.batch])
            else:
                n=torch.from_numpy(np.array(noise[start:start+rt.batch])).cuda()
                l=torch.from_numpy(np.array(labels[start:start+rt.batch])).cuda()
                torch.cuda.synchronize();beg=time.perf_counter()
                try:
                    z,stats=rt.sample(n,l,point['method'],point['alpha'],diagnostics=start==first)
                except (FloatingPointError,AssertionError) as error:
                    if not isinstance(error,FloatingPointError) and 'underflow in dt' not in str(error):
                        raise
                    failure=dict(model=name,**point,rank=rank,start=start,error=repr(error),
                                 complete=False,fid=None,request_sha256=rh)
                    atomic(d/'numerical_failure.json',failure)
                    try:
                        os.link(d/'numerical_failure.json',out/point['arm']/'numerical_failure.json')
                    except FileExistsError:
                        pass
                    break
                torch.cuda.synchronize();trajectory=time.perf_counter()-beg
                beg=time.perf_counter();pix=rt.decode(z);torch.cuda.synchronize();decode=time.perf_counter()-beg
                assert pix.shape==(rt.batch,256,256,3) and pix.dtype==np.uint8
                tmp=p.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    np.savez(stream,arr_0=pix,labels=np.array(labels[start:start+rt.batch]),
                             noise_sha256=array_sha(noise[start:start+rt.batch]),request_sha256=rh,
                             trajectory_seconds=trajectory,decode_seconds=decode,
                             full_calls=stats['counts']['full'],prefix_calls=stats['counts']['prefix'],
                             endpoint_rms=np.array(stats['endpoint_rms']))
                tmp.replace(p)
                if start==first:atomic(d/'diagnostics.json',stats)
            files.append(dict(start=start,file=p.name,sha256=sha(p)))
            if len(files)==1 or len(files)%8==0:
                value=dict(model=name,arm=point['arm'],rank=rank,images=len(files)*rt.batch)
                atomic(d/'progress.json',value);print(json.dumps(value),flush=True)
        failed=(out/point['arm']/'numerical_failure.json').exists()
        atomic(d/'summary.json',dict(complete=not failed,rank=rank,files=files,request_sha256=rh,
                                    noise_sha256=bank['noise_sha256'],label_sha256=bank['label_sha256']))
        while not (out/point['arm']/'advance.json').exists():time.sleep(.25)


def wait_files(paths,workers):
    while not all(p.exists() for p in paths):
        codes=[p.poll() for p in workers]
        if any(code is not None for code in codes):
            raise RuntimeError(f'Worker exited before required files: {codes}')
        time.sleep(.5)


def evaluate(name,point,spec):
    out=ROOT/name/point['arm'];bank=spec['bank'];batch=bank['batch'];rh=sha(ROOT/'request.json')
    if (out/'numerical_failure.json').exists():
        result=dict(model=name,**point,reused=False,complete=False,fid=None,
                    status='numerical_failure',failure=read(out/'numerical_failure.json'),
                    completed_images=sum(len(read(out/f'rank{r}/summary.json')['files'])*batch for r in range(4)),
                    request_sha256=rh)
        atomic(out/'result.json',result)
        return result
    images=np.empty((1000,256,256,3),np.uint8);seen=set();trajectory=decode=0.;full=prefix=0
    labels=np.load(ROOT/name/'labels.npy',mmap_mode='r')
    for rank in range(4):
        d=out/f'rank{rank}';summary=read(d/'summary.json')
        assert summary['complete'] and summary['request_sha256']==rh
        assert summary['noise_sha256']==bank['noise_sha256'] and summary['label_sha256']==bank['label_sha256']
        for rec in summary['files']:
            start=rec['start'];p=d/rec['file'];assert start not in seen and (start//batch)%4==rank
            seen.add(start);assert sha(p)==rec['sha256']
            with np.load(p) as b:
                assert str(b['request_sha256'])==rh
                np.testing.assert_array_equal(b['labels'],labels[start:start+batch])
                images[start:start+batch]=b['arr_0'];trajectory+=float(b['trajectory_seconds']);decode+=float(b['decode_seconds'])
                full+=int(b['full_calls']);prefix+=int(b['prefix_calls'])
    assert seen==set(range(0,1000,batch))
    np.savez(out/'samples.npz',arr_0=images)
    if name=='sit_small':
        cmd=['/data/shared/envs/adm-fid/bin/python','experiments/compute_adm_fid.py',
             '--reference',str(SMALL_DATA/'adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'),
             '--samples',str(out/'samples.npz'),'--output',str(out/'fid.json'),
             '--activations-output',str(out/'activations.npz'),'--batch-size','8','--gpu-memory-fraction','.30']
    else:
        cmd=[sys.executable,'experiments/evaluate_raev2_official_samples.py','--branch',name+'_'+point['arm']+'='+str(out/'samples.npz'),
             '--output',str(out/'fid.csv'),'--batch-size','32','--device','cuda','--feature-cache-dir',str(out/'features')]
    with (out/'evaluation.log').open('w') as f:
        subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT,check=True)
    metrics=read(out/'fid.json');metrics=metrics[0] if isinstance(metrics,list) else metrics
    result=dict(model=name,**point,reused=False,complete=True,samples=1000,fid=metrics['fid'],metrics=metrics,
                noise_sha256=bank['noise_sha256'],label_sha256=bank['label_sha256'],
                sample_path=str(out/'samples.npz'),sample_sha256=sha(out/'samples.npz'),
                trajectory_gpu_seconds=trajectory,decode_gpu_seconds=decode,sum_batch_gpu_seconds=trajectory+decode,
                full_calls_per_image=full/(1000/batch),prefix_calls_per_image=prefix/(1000/batch),request_sha256=rh)
    atomic(out/'result.json',result)
    return result


def run():
    request=read(ROOT/'request.json')
    assert read(ROOT/'status.json')['phase']=='prepared', 'Inspect any previous run before resuming'
    for p,h in request['sources'].items():assert sha(p)==h,p
    results=[];begin=time.perf_counter()
    for name in request['model_order']:
        spec=request['models'][name];out=ROOT/name
        for p,h in spec['assets'].items():assert sha(p)==h,p
        for rec in spec['reused'].values():
            assert sha(rec['sample_path'])==rec['sample_sha256']
            for p,h in rec['source_files'].items():assert sha(p)==h,p
        workers=[];logs=[]
        try:
            for rank in range(4):
                f=(out/f'worker{rank}.log').open('w');logs.append(f)
                workers.append(subprocess.Popen([sys.executable,'-m','experiments.run_lifting_scale_sweep_20260909',
                    '--model',name,'--rank',str(rank)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT))
            atomic(ROOT/'status.json',dict(phase='preflight',model=name,pids=[p.pid for p in workers],research_goal_achieved=False))
            wait_files([out/f'preflight_rank{rank}.json' for rank in range(4)],workers)
            checks=[read(out/f'preflight_rank{rank}.json') for rank in range(4)]
            assert all(c['passed'] for c in checks)
            assert all(c['runtime_sources']==checks[0]['runtime_sources'] for c in checks)
            atomic(out/'preflight_passed.json',dict(passed=True,checks=checks))
            results.extend(spec['reused'].values());atomic(ROOT/'results.json',results)
            for point in spec['new_points']:
                atomic(ROOT/'status.json',dict(phase='sampling',model=name,arm=point['arm'],pids=[p.pid for p in workers],research_goal_achieved=False))
                wait_files([out/point['arm']/f'rank{rank}/summary.json' for rank in range(4)],workers)
                atomic(ROOT/'status.json',dict(phase='evaluating',model=name,arm=point['arm'],pids=[p.pid for p in workers],research_goal_achieved=False))
                r=evaluate(name,point,spec);results.append(r);atomic(ROOT/'results.json',results)
                print(json.dumps(r),flush=True)
                atomic(out/point['arm']/'advance.json',dict(complete=True))
            codes=[p.wait() for p in workers];assert codes==[0]*4,codes
            atomic(out/'complete.json',dict(all_points_processed=True,points=len(spec['new_points']),
                                           reused_points=len(spec['reused']),
                                           numerical_failures=sum(r['model']==name and not r['complete'] for r in results)))
        finally:
            for p in workers:
                if p.poll() is None:p.terminate()
            for p in workers:p.wait()
            for f in logs:f.close()
    for p,h in request['sources'].items():assert sha(p)==h,p
    atomic(ROOT/'status.json',dict(phase='complete',wall_seconds=time.perf_counter()-begin,
                                   research_goal_achieved=False,results=len(results),
                                   numerical_failures=sum(not r['complete'] for r in results)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run-prepared',action='store_true')
    p.add_argument('--rank',type=int,choices=range(4));p.add_argument('--model',choices=MODELS);args=p.parse_args()
    if args.prepare:
        prepare()
    elif args.rank is not None:
        assert args.model
        worker(args.model,args.rank)
    elif args.run_prepared:
        try:run()
        except BaseException as error:
            atomic(ROOT/'status.json',dict(phase='failed',error=repr(error),research_goal_achieved=False));raise
    else:p.error('Choose --prepare, --run-prepared, or --model with --rank')
