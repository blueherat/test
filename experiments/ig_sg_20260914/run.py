"""Frozen paired IG + SG experiment; three provenance-preserving RAE arm reuses."""
import argparse,fcntl,gc,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.ig_sg_20260914 import core as c
from experiments.weak_reference_20260914 import cross_run as prior
from experiments.weak_reference_20260914.cross_run import save_npz,collect,pop_job

MODULE='experiments.ig_sg_20260914.run'
OLD=c.common.EXPS/'weak_reference_20260914'
SIT_REF=c.common.DATA/'imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'


def sources(name):
    files=[Path(__file__),Path(c.__file__),Path(__file__).with_name('check.py'),Path(prior.__file__),
        c.common.WORK/'experiments/weak_reference_20260914/sampler.py',
        c.common.WORK/'experiments/self_guidance_20260913/sampler.py']
    files+=c.common.source_paths(name)
    files+=[Path(p) for p in c.common.read(c.ROOT/'checks'/f'{name}.json')['runtime_sources']]
    if name=='sit_small':
        files += [c.common.WORK/'experiments/compute_adm_fid.py',c.common.WORK/'train_gen/evaluator.py']
    else:files+=prior.sources(name)
    return sorted(set(p.resolve() for p in files))


def prepare(args):
    for name in args.models.split(','):
        root=c.ROOT/args.phase/name;root.mkdir(parents=True,exist_ok=True)
        check=c.common.read(c.ROOT/'checks'/f'{name}.json');assert check['model_passed']
        old=OLD/'sit_screen_1k' if name=='sit_small' else OLD/'cross_screen_1k'/name
        oldrequest=c.common.read(old/'request.json')
        assert oldrequest['seed']==2026091407 and oldrequest['samples']==1000
        if name=='sit_small':
            assert c.common.sha(old/'inputs.npz')==oldrequest['inputs_sha256']
            with np.load(old/'inputs.npz') as bank:
                for key in ('noise','labels'):
                    assert c.common.array_sha(bank[key])==oldrequest[key+'_sha256']
                    if not (root/f'{key}.npy').exists():np.save(root/f'{key}.npy',bank[key])
                    assert np.array_equal(np.load(root/f'{key}.npy'),bank[key])
            origin={str(old/'inputs.npz'):c.common.sha(old/'inputs.npz')}
        else:
            prior.verify(old)
            origin=oldrequest['inputs']
            for key in ('noise','labels'):
                dest=root/f'{key}.npy'
                if not dest.exists():shutil.copy2(old/f'{key}.npy',dest)
                assert c.common.sha(dest)==c.common.sha(old/f'{key}.npy')
        assets={**oldrequest['assets']}
        for p in c.common.asset_paths(name):assets[str(p)]=c.common.sha(p)
        if name=='sit_small':
            vae=Path('/home/zhoushunyu/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse')
            extras=list((vae/'blobs').glob('*'))+[vae/'refs/main',Path('/data/shared/adm_refs/classify_image_graph_def.pb')]
            assert len(extras)>=4
            for p in extras:assets[str(p)]=c.common.sha(p)
        reused={}
        if name=='raev2':
            assert check['raev2_prior_implementation_bitwise']
            for arm,oldarm in {'ig':'ig','ig_log_k02_w1':'ig_log_k02_w1','ig_log_cost':'ig_equal_nfe'}.items():
                out=old/oldarm
                summary=c.common.read(out/'summary.json')
                assert summary['complete'] and summary['samples']==1000
                assert summary['request_sha256']==c.common.sha(old/'request.json')
                reused[arm]=dict(out=str(out),request_sha256=c.common.sha(old/'request.json'),
                    summary_sha256=c.common.sha(out/'summary.json'),metrics_sha256=c.common.sha(out/'metrics.json'))
        request=dict(model=name,phase=args.phase,samples=1000,seed=2026091407,batch=16 if name=='sit_small' else 32,
            configs=c.configs(name),sources={str(p):c.common.sha(p) for p in sources(name)},assets=assets,
            inputs={str(root/f'{k}.npy'):c.common.sha(root/f'{k}.npy') for k in ('noise','labels')},
            input_origin=origin,reused_arms=reused,checks_sha256=c.common.sha(c.ROOT/'checks'/f'{name}.json'),
            reference=str(SIT_REF if name=='sit_small' else prior.REFERENCE),
            ig_schedule='SiT: alpha .8*6/7 for t<.25, .8 for t<.5; JiT: .3 for t<.5; RAE: weak+1.78*(full-weak) for .1<=t<=1',
            precision='FP32 weights/state; SiT no autocast; JiT/RAE BF16 autocast; TF32 enabled',
            definition='IG field plus raw strong residual. Paper SG shift .01 toward noisy time. Log SG antithetic probes kappa .2 times path noise std. omega 1. No CFG.',
            cost='Observed full transformer traversals; weak head shares trunk. SiT SG 255 vs control256; other controls exact NFE. GPU batch sampling/decode seconds, evaluation excluded.',
            scope='Paired 1K pilot, fixed prior coefficients, same seed as prior screen; reused arms are not new independent replications.')
        path=root/'request.json'
        if path.exists():assert c.common.read(path)==request,'Frozen phase changed'
        else:c.common.atomic(path,request)
        print('prepared',name,'new',5-len(reused),'reuse',len(reused),flush=True)


def verify(root):
    request=c.common.read(root/'request.json')
    for group in ('sources','assets','inputs','input_origin'):
        for path,sha in request[group].items():assert c.common.sha(path)==sha,(group,path)
    assert c.common.sha(c.ROOT/'checks'/f"{request['model']}.json")==request['checks_sha256']
    for item in request['reused_arms'].values():
        out=Path(item['out'])
        for key,path in [('request',out.parent/'request.json'),('summary',out/'summary.json'),('metrics',out/'metrics.json')]:
            assert c.common.sha(path)==item[key+'_sha256']
    return request


def evaluate(out,name):
    if (out/'metrics.json').exists():return
    if name!='sit_small':return prior.evaluate(out)
    command=['/data/shared/envs/adm-fid/bin/python',str(c.common.WORK/'experiments/compute_adm_fid.py'),
        '--reference',str(SIT_REF),'--samples',str(out/'samples.npz'),'--batch-size','32',
        '--gpu-memory-fraction','.3','--output',str(out/'fid.json'),
        '--activations-output',str(out/'inception_activations.npz')]
    with (out/'evaluation.log').open('w') as stream:
        subprocess.run(command,check=True,cwd=c.common.WORK,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    c.common.atomic(out/'metrics.json',[c.common.read(out/'fid.json')])


@torch.inference_mode()
def worker(args):
    root=c.ROOT/args.phase;rt=None
    while (job:=pop_job(root)) is not None:
        name,config=job['model'],job['config'];model_root=root/name;request=verify(model_root)
        if rt is None or rt.name!=name:
            if rt is not None:rt.close()
            del rt;gc.collect();torch.cuda.empty_cache();rt=c.Runtime(name)
        noise=np.load(model_root/'noise.npy',mmap_mode='r');labels=np.load(model_root/'labels.npy')
        out=model_root/config['arm'];out.mkdir(parents=True,exist_ok=True)
        if not (out/'summary.json').exists():
            c.sample(rt,torch.from_numpy(noise[:2].copy()).cuda(),torch.from_numpy(labels[:2]).cuda(),config)
            for start in range(0,request['samples'],request['batch']):
                path=out/f'batch{start:06d}.npz'
                if path.exists():continue
                stop=min(start+request['batch'],request['samples'])
                z=torch.from_numpy(noise[start:stop].copy()).cuda();y=torch.from_numpy(labels[start:stop]).cuda()
                torch.cuda.synchronize();begin=time.perf_counter()
                z,counts=c.sample(rt,z,y,config,trace=start==0);pixels=rt.decode(z)
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                save_npz(path,pixels=pixels,latents=z.float().cpu().numpy(),labels=labels[start:stop],
                    start=start,full=counts['full'],seconds=elapsed,trace=counts['trace'],
                    request_sha256=c.common.sha(model_root/'request.json'),noise_sha256=c.common.array_sha(noise[start:stop]))
                c.common.atomic(root/f'progress{args.rank}.json',dict(model=name,arm=config['arm'],completed=stop,total=1000,phase='sampling'))
            collect(out,request,noise,labels)
        torch.cuda.empty_cache()
        c.common.atomic(root/f'progress{args.rank}.json',dict(model=name,arm=config['arm'],phase='evaluation'))
        evaluate(out,name);print(name,config['arm'],c.common.read(out/'metrics.json')[0]['fid'],flush=True)
    if rt is not None:rt.close()
    c.common.atomic(root/f'worker{args.rank}_complete.json',dict(complete=True))


def controller(args):
    root=c.ROOT/args.phase
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);jobs=[]
        for name in args.models.split(','):
            request=verify(root/name)
            for config in request['configs']:
                if config['arm'] not in request['reused_arms'] and not (root/name/config['arm']/'metrics.json').exists():
                    jobs.append(dict(model=name,config=config))
        weight={'raev2':8,'jit':2,'sit_small':.2}
        jobs.sort(key=lambda j:weight[j['model']]*j['config']['steps']*({'baseline':1,'sg':2,'log':3}[j['config']['kind']]),reverse=True)
        c.common.atomic(root/'queue.json',jobs);workers=[];logs=[]
        for rank,gpu in enumerate(args.gpus.split(',')):
            stream=(root/f'worker{rank}.log').open('a');logs.append(stream)
            workers.append(subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--phase',args.phase,'--rank',str(rank)],
                cwd=c.common.WORK,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')))
        while any(p.poll() is None for p in workers):
            codes=[p.poll() for p in workers]
            if any(code not in (None,0) for code in codes):
                for p in workers:
                    if p.poll() is None:p.terminate()
                for p in workers:p.wait()
                break
            c.common.atomic(root/'status.json',dict(complete=False,new_completed=len(list(root.glob('*/*/metrics.json'))),
                remaining_queue=len(c.common.read(root/'queue.json')),workers=[p.pid for p in workers]))
            time.sleep(5)
        codes=[p.wait() for p in workers]
        for stream in logs:stream.close()
        c.common.atomic(root/'controller_complete.json',dict(complete=all(code==0 for code in codes),exit_codes=codes))
        if any(codes):raise RuntimeError(codes)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','controller','worker'])
    p.add_argument('--phase',default='screen_1k');p.add_argument('--models',default='sit_small,jit,raev2')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--gpus',default='0,1,2,3')
    args=p.parse_args();globals()[args.action](args)

if __name__=='__main__':main()
