"""Resumable fusion tuning, followed by selected configurations on new 5K noise."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import numpy as np
import torch
from experiments import sit_guidance_fusion_20260910 as fusion
from experiments import small_sit_carrier_flow_20260909 as infrastructure
from experiments.sit_guidance_portfolio_20260910 import runner as parent
from experiments.lifting_scale_sweep_20260909 import EXPS,WORK,array_sha,atomic,read,sha

ROOT=EXPS/'sit_guidance_fusion_20260910'
INITIAL=EXPS/'sit_guidance_portfolio_confirmation_20260910'
MODULE='experiments.sit_guidance_fusion_pipeline_20260910'
PYTHON=parent.assets.PYTHON
PROTOCOL=WORK/'docs/SIT_GUIDANCE_FUSION_PROTOCOL_20260910_ZH.md'
STAGES={'tuning_1k':(1000,202610062),'selected_5k':(5000,202610064)}
BATCH,RANKS=8,4


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def additional_sources():
    return [Path(__file__).resolve(),Path(fusion.__file__).resolve(),Path(fusion.angular.__file__).resolve(),
        WORK/'experiments/analyze_sit_guidance_followup_20260910.py',PROTOCOL]


def verify_parent():
    request=read(parent.ROOT/'request.json')
    for category in ('sources','assets'):
        for path,digest in request[category].items():assert sha(path)==digest,path
    for name,digest in request['bank_files'].items():assert sha(parent.BANK_ROOT/name)==digest,name
    parent.assets.verify()
    assert sha(infrastructure.REFERENCE)==request['reference_sha256']
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb')==request['inception_graph_sha256']
    return request


def golden_pairs(configs):
    old_records=read(parent.ROOT/'results.json');pairs=[]
    for config in configs:
        candidates=[r for r in old_records if all(r[k]==config[k] for k in ('key','source','strength','theta','solver','cutoff'))]
        if candidates:pairs.append((config,min(candidates,key=lambda r:r['fid'])))
    return pairs


@torch.inference_mode()
def development_check():
    cpu=fusion.cpu_checks();verify_parent();ROOT.mkdir(parents=True,exist_ok=True)
    configs=fusion.configurations();fusion.install();rt=parent.operators.make_runtime()
    noise=torch.from_numpy(np.load(parent.BANK_ROOT/'noise.npy')[:8].copy()).cuda()
    labels=torch.from_numpy(np.load(parent.BANK_ROOT/'labels.npy')[:8].copy()).cuda()
    limits=fusion.limiting_checks(rt,noise,labels)
    rt.labels=labels;hooks=parent.hook_counts(rt)
    original,_=fusion.sample(rt,noise,labels,configs[0])
    fields=[];trajectories=[]
    for config in configs:
        if config['key'] not in ('composed_guidance','paper_adg'):continue
        for tv in (0.,config['cutoff']-1/64):
            ctx=parent.operators.StepContext(shadow=noise.clone())
            value,info=fusion.evaluate(rt,noise,tv,config,parent.operators.amount_at(config,tv),ctx)
            assert torch.isfinite(value).all() and torch.isfinite(info['diagnostics']).all(),config['arm']
            assert parent.hook_counts(rt)==hooks and rt.labels is labels
        fields.append(config['arm'])
    for family in (*fusion.FAMILIES,'ig_apg','ig_adg'):
        config=[c for c in configs if c['family']==family][-1]
        latent,stats=fusion.sample(rt,noise,labels,config)
        assert torch.isfinite(latent).all() and np.isfinite(stats['diagnostics']).all()
        zero,_=fusion.sample(rt,noise,labels,config,zero=True)
        assert torch.equal(zero,original),family
        trajectories.append(dict(arm=config['arm'],family=family,full=stats['full_calls'],prefix=stats['prefix_calls'],
            auxiliary=stats['auxiliary_full_calls'],max_abs=float(latent.abs().max())))
        print(json.dumps(dict(preflight_trajectory=family,full=stats['full_calls'],prefix=stats['prefix_calls'])),flush=True)
    # All native/component settings shared with the original portfolio must
    # reproduce it. Different JSON arm names do not change the random stream.
    golden=[]
    representatives={}
    for config,row in golden_pairs(configs):representatives.setdefault(config['family'],(config,row))
    for config,row in representatives.values():
        latent,_=fusion.sample(rt,noise,labels,config)
        path=parent.ROOT/row['arm']/'rank0/batch0000.npz'
        with np.load(path) as batch:np.testing.assert_array_equal(latent.cpu().numpy(),batch['latents'])
        golden.append(dict(arm=config['arm'],old_arm=row['arm'],path=str(path),sha256=sha(path),exact=True))
    repeated,_=fusion.sample(rt,noise,labels,configs[0]);assert torch.equal(original,repeated)
    result=dict(passed=True,cpu=cpu,limiting_components=limits,grid_arms=fields,trajectories=trajectories,
        golden=golden,zero_trajectories_exact=True,native_after_candidates_exact=True,
        runtime_sources=rt.sources,source_hashes={str(path):sha(path) for path in additional_sources()},
        no_fid_used=True)
    atomic(ROOT/'development_check.json',result)
    print(json.dumps(dict(development_check_passed=True,grid=len(fields),trajectories=len(trajectories))),flush=True)


def prepare_stage(stage,configs,selection=None):
    base=ROOT/stage
    if (base/'request.json').exists():
        existing=read(base/'request.json');assert existing['configs']==configs
        return
    assert not base.exists(),base
    parent_request=verify_parent();check=read(ROOT/'development_check.json');assert check['passed']
    for path,digest in check['source_hashes'].items():assert sha(path)==digest,path
    initial_status=read(INITIAL/'status.json');assert initial_status['phase']=='complete'
    samples,seed=STAGES[stage]
    expected_bytes=len(configs)*samples*(256*256*3*2+4*32*32*8+32768)
    assert shutil.disk_usage(ROOT).free>expected_bytes+(10<<30)
    base.mkdir();bank=base/'inputs';bank.mkdir()
    generator=torch.Generator(device='cuda').manual_seed(seed)
    noise=np.lib.format.open_memmap(bank/'noise.npy',mode='w+',dtype=np.float32,shape=(samples,4,32,32))
    for start in range(0,samples,8):
        noise[start:start+8]=torch.randn((8,4,32,32),device='cuda',generator=generator).cpu().numpy()
    noise.flush()
    labels=np.random.default_rng(seed+1).permutation(np.repeat(np.arange(100,dtype=np.int64),samples//100))
    np.save(bank/'labels.npy',labels)
    source_hashes=dict(parent_request['sources']);source_hashes.update(check['source_hashes'])
    snapshot=base/'sources';snapshot.mkdir()
    for i,path in enumerate(sorted(source_hashes)):(snapshot/f'{i:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
    references={str(path):sha(path) for path in [ROOT/'development_check.json',INITIAL/'request.json',
        INITIAL/'results.json',parent.ROOT/'request.json',parent.BANK_ROOT/'noise.npy',parent.BANK_ROOT/'labels.npy']}
    if stage=='selected_5k':
        for path in [ROOT/'tuning_1k/request.json',ROOT/'tuning_1k/results.json',ROOT/'tuning_1k/selection.json']:
            references[str(path)]=sha(path)
    original_noise=read(INITIAL/'request.json')['bank']['noise_sha256']
    assert array_sha(noise) not in (original_noise,parent_request['bank']['noise_sha256'])
    if stage=='selected_5k':assert array_sha(noise)!=read(ROOT/'tuning_1k/request.json')['bank']['noise_sha256']
    request=dict(stage=stage,configs=configs,samples=samples,batch=8,ranks=4,arms=[c['arm'] for c in configs],
        sources=source_hashes,assets=parent_request['assets'],reference=str(infrastructure.REFERENCE),
        reference_sha256=parent_request['reference_sha256'],inception_graph_sha256=parent_request['inception_graph_sha256'],
        references=references,bank_root=str(bank),bank_files={name:sha(bank/name) for name in ('noise.npy','labels.npy')},
        bank=dict(samples=samples,batch=8,noise_seed=seed,label_seed=seed+1,noise_sha256=array_sha(noise),
            label_sha256=array_sha(labels),classes_balanced=True,noise_generation='continuous CUDA generator; sequential B8'),
        independent_confirmation=stage=='selected_5k',selection=selection,
        family_definitions=fusion.FAMILIES,diagnostic_names=parent.DIAGNOSTICS,
        python=PYTHON,estimated_output_bytes=expected_bytes,prepared_unix=time.time())
    atomic(base/'request.json',request);atomic(base/'status.json',dict(phase='prepared',completed=0,total=len(configs)))
    print(json.dumps(dict(prepared_stage=stage,arms=len(configs),samples=samples,request_sha256=sha(base/'request.json'))),flush=True)


def verify_stage(stage):
    base=ROOT/stage;request=read(base/'request.json')
    for category in ('sources','assets','references'):
        for path,digest in request[category].items():assert sha(path)==digest,path
    for name,digest in request['bank_files'].items():assert sha(base/'inputs'/name)==digest,name
    assert sha(infrastructure.REFERENCE)==request['reference_sha256']
    assert sha('/data/shared/adm_refs/classify_image_graph_def.pb')==request['inception_graph_sha256']
    return request,sha(base/'request.json')


def verify_batch(path,receipt,config,start,h,noise,labels):
    record=read(receipt)
    assert record['request_sha256']==h and record['config_sha256']==canonical_hash(config)
    assert record['start']==start and record['sha256']==sha(path),path
    with np.load(path) as batch:
        assert str(batch['request_sha256'])==h
        assert str(batch['noise_sha256'])==array_sha(noise[start:start+8])
        np.testing.assert_array_equal(batch['labels'],labels[start:start+8])
        assert batch['latents'].shape==(8,4,32,32) and np.isfinite(batch['latents']).all()
        assert batch['arr_0'].shape==(8,256,256,3) and batch['arr_0'].dtype==np.uint8
    return dict(start=start,file=path.name,sha256=record['sha256'])


def ensure_parent(pid):
    if os.getppid()!=pid:raise RuntimeError('Controller exited; stopping worker.')


def worker(stage,rank,run_id,parent_pid):
    request,h=verify_stage(stage);base=ROOT/stage;directory=base/'runs'/run_id
    configs={c['arm']:c for c in request['configs']};noise=np.load(base/'inputs/noise.npy',mmap_mode='r')
    labels=np.load(base/'inputs/labels.npy',mmap_mode='r');samples=request['samples']
    fusion.install();rt=parent.operators.make_runtime();cuda=lambda x:torch.from_numpy(np.array(x)).cuda()
    n=cuda(np.load(parent.BANK_ROOT/'noise.npy')[rank*8:rank*8+8]);y=cuda(np.load(parent.BANK_ROOT/'labels.npy')[rank*8:rank*8+8])
    rt.labels=y
    for tv in (0.,.375):
        t=n.new_tensor(tv);s,w=rt.pair(n,t)
        assert torch.equal(s,rt.field(n,t,'full')) and torch.equal(w,rt.field(n,t,'base'))
    for path,digest in rt.sources.items():assert request['sources'].get(path)==digest,path
    fusion.limiting_checks(rt,n,y)
    anchor=next(c for c in fusion.configurations() if c['family']=='ig_local' and c['strength']==.8)
    z,_=fusion.sample(rt,n,y,anchor)
    with np.load(parent.ROOT/f'i40_ig_local_attention_g2_p1/rank{rank}/batch{rank*8:04d}.npz') as batch:
        np.testing.assert_array_equal(z.cpu().numpy(),batch['latents'])
    atomic(directory/f'ready{rank}.json',dict(passed=True,rank=rank,request_sha256=h,run_id=run_id,
        pid=os.getpid(),native_prefix_exact=True,fusion_limits_exact=True,old_local_ig_exact=True))
    previous=None
    while True:
        ensure_parent(parent_pid)
        path=directory/'command.json';command=read(path) if path.exists() else {}
        if command.get('command')=='complete':return
        if command.get('command')!='sample' or command.get('arm')==previous:
            time.sleep(.5);continue
        config=configs[command['arm']];arm=config['arm'];out=base/arm/f'rank{rank}';out.mkdir(parents=True,exist_ok=True)
        files=[]
        for start in range(rank*8,samples,32):
            ensure_parent(parent_pid)
            if (base/arm/'numerical_failure.json').exists():break
            path=out/f'batch{start:04d}.npz';receipt=path.with_suffix('.receipt.json')
            if path.exists() and receipt.exists():
                files.append(verify_batch(path,receipt,config,start,h,noise,labels));continue
            if path.exists() or receipt.exists():
                orphan=out/'orphans'/run_id;orphan.mkdir(parents=True,exist_ok=True)
                for part in (path,receipt):
                    if part.exists():part.replace(orphan/part.name)
            n,y=cuda(noise[start:start+8]),cuda(labels[start:start+8])
            torch.cuda.synchronize();begin=time.perf_counter()
            try:z,stats=fusion.sample(rt,n,y,config)
            except (FloatingPointError,AssertionError) as error:
                if not isinstance(error,FloatingPointError) and 'underflow in dt' not in str(error):raise
                failure=dict(arm=arm,rank=rank,start=start,error=repr(error),request_sha256=h)
                atomic(out/'numerical_failure.json',failure)
                try:os.link(out/'numerical_failure.json',base/arm/'numerical_failure.json')
                except FileExistsError:pass
                break
            torch.cuda.synchronize();trajectory=time.perf_counter()-begin;begin=time.perf_counter()
            pixels=rt.decode(z);torch.cuda.synchronize();decode=time.perf_counter()-begin
            assert pixels.shape==(8,256,256,3) and pixels.dtype==np.uint8
            temp=path.with_suffix('.npz.tmp')
            with temp.open('wb') as stream:
                np.savez(stream,arr_0=pixels,latents=z.cpu().numpy(),labels=np.array(labels[start:start+8]),
                    request_sha256=h,noise_sha256=array_sha(noise[start:start+8]),
                    trajectory_seconds=trajectory,decode_seconds=decode,**stats)
                stream.flush();os.fsync(stream.fileno())
            temp.replace(path)
            atomic(receipt,dict(request_sha256=h,config_sha256=canonical_hash(config),start=start,sha256=sha(path)))
            files.append(verify_batch(path,receipt,config,start,h,noise,labels))
            progress=dict(arm=arm,rank=rank,images=len(files)*8,run_id=run_id)
            atomic(out/'progress.json',progress)
            if len(files)==1 or len(files)%16==0:print(json.dumps(progress),flush=True)
        atomic(out/'summary.json',dict(complete=not (base/arm/'numerical_failure.json').exists(),
            rank=rank,files=files,request_sha256=h,run_id=run_id,
            noise_sha256=request['bank']['noise_sha256'],label_sha256=request['bank']['label_sha256']))
        previous=arm


def committed(base,config,h):
    directory=base/config['arm'];path=directory/'commit.json'
    if not path.exists():return None
    commit=read(path);assert commit['request_sha256']==h and commit['config_sha256']==canonical_hash(config)
    for name,digest in commit['files'].items():assert sha(directory/name)==digest,directory/name
    result=read(directory/'result.json');assert result['request_sha256']==h
    return result


def run_stage(stage):
    from experiments.analyze_sit_guidance_followup_20260910 import write_progress
    base=ROOT/stage;lock=(base/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request,h=verify_stage(stage);configs=request['configs'];results=[];workers=[];streams=[]
    run_id=time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8];directory=base/'runs'/run_id;directory.mkdir(parents=True)
    begin=time.perf_counter()
    infrastructure.ROOT,infrastructure.BANK_ROOT=base,base/'inputs'
    infrastructure.SAMPLES,infrastructure.BATCH,infrastructure.RANKS=request['samples'],8,4
    def status(phase,**extra):
        value=dict(stage=stage,phase=phase,completed=len(results),total=len(configs),run_id=run_id,
            controller_pid=os.getpid(),worker_pids=[p.pid for p in workers],run_wall_seconds=time.perf_counter()-begin,**extra)
        atomic(base/'status.json',value);atomic(ROOT/'status.json',value)
    def wait_for(paths):
        while True:
            codes=[p.poll() for p in workers]
            if any(code is not None for code in codes):raise RuntimeError(f'Worker exited early: {codes}')
            if all(p.exists() and read(p).get('run_id')==run_id for p in paths):return
            time.sleep(.5)
    try:
        for config in configs:
            row=committed(base,config,h)
            if row is not None:results.append(row)
        done={r['arm'] for r in results};atomic(base/'results.json',results);write_progress(base,request,results)
        if len(done)==len(configs):status('complete');return
        for rank in range(4):
            stream=(directory/f'worker{rank}.log').open('a');streams.append(stream)
            workers.append(subprocess.Popen([PYTHON,'-u','-m',MODULE,'--worker',str(rank),
                '--stage',stage,'--run-id',run_id,'--parent-pid',str(os.getpid())],cwd=WORK,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT))
        status('preflight');wait_for([directory/f'ready{r}.json' for r in range(4)])
        checks=[read(directory/f'ready{r}.json') for r in range(4)];assert all(c['passed'] for c in checks)
        atomic(directory/'preflight_passed.json',dict(passed=True,checks=checks,request_sha256=h))
        for config in configs:
            arm=config['arm']
            if arm in done:continue
            if (ROOT/'STOP_AFTER_CURRENT').exists():break
            if shutil.disk_usage(base).free<(2<<30):raise RuntimeError('Less than 2 GiB free; preserving existing output and stopping.')
            for path,digest in request['sources'].items():assert sha(path)==digest,path
            status('sampling',arm=arm,family=config['family']);atomic(directory/'command.json',dict(command='sample',arm=arm))
            wait_for([base/arm/f'rank{r}/summary.json' for r in range(4)])
            status('evaluating',arm=arm,family=config['family']);eval_begin=time.perf_counter()
            result=infrastructure.evaluate(arm,request,h);result.update(config)
            result['evaluation_wall_seconds']=time.perf_counter()-eval_begin;atomic(base/arm/'result.json',result)
            names=['result.json']+(['samples.npz','latents.npy','activations.npz','fid.json'] if result['complete'] else ['numerical_failure.json'])
            atomic(base/arm/'commit.json',dict(request_sha256=h,config_sha256=canonical_hash(config),
                files={name:sha(base/arm/name) for name in names}))
            results.append(result);done.add(arm);atomic(base/'results.json',results);write_progress(base,request,results)
            print(json.dumps(dict(stage=stage,arm=arm,fid=result['fid'],complete=result['complete'],completed=len(results),total=len(configs))),flush=True)
        atomic(directory/'command.json',dict(command='complete'))
        codes=[p.wait(timeout=45) for p in workers];assert codes==[0]*4,codes
        status('complete' if len(results)==len(configs) else 'stopped_after_current',
            worker_exit_codes=codes,numerical_failures=sum(not r['complete'] for r in results))
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        for p in workers:
            if p.poll() is None:p.terminate()
        for p in workers:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()
        lock.close()


def choose_final():
    base=ROOT/'tuning_1k';request=read(base/'request.json');rows=read(base/'results.json')
    assert read(base/'status.json')['phase']=='complete'
    selected={};decisions=[]
    for source in ('ig','cfg'):
        pool=[r for r in rows if r['complete'] and r['source']==source]
        candidates=[r for r in pool if r['role']=='fusion']
        candidate=min(candidates,key=lambda r:r['fid']) if candidates else None
        component=min((r for r in pool if r['role']!='fusion'),key=lambda r:r['fid'])
        native=min((r for r in pool if r['role']=='native' and r['family']!='strong'),key=lambda r:r['fid'])
        eligible=candidate is not None and candidate['fid']<component['fid']
        decisions.append(dict(source=source,best_fusion=candidate,best_component=component,best_native=native,
            confirm=eligible,rule='fusion must beat the best retuned non-fusion component on the new 1K bank'))
        if eligible:
            for row in (native,component,candidate):selected[row['arm']]=next(c for c in request['configs'] if c['arm']==row['arm'])
            if source=='ig':
                for family in ('ig_native','ig_dopri'):
                    row=min((r for r in pool if r['family']==family),key=lambda r:r['fid'])
                    selected[row['arm']]=next(c for c in request['configs'] if c['arm']==row['arm'])
    result=dict(decisions=decisions,configs=list(selected.values()),tuning_request_sha256=sha(base/'request.json'),
        tuning_results_sha256=sha(base/'results.json'),independent_final_bank_seed=202610064,
        boundary_winners_are_fixed_config_confirmations_not_claims_of_global_optimality=True)
    path=base/'selection.json'
    if path.exists():assert read(path)==result
    else:atomic(path,result)
    return result


def pipeline():
    from experiments.analyze_sit_guidance_followup_20260910 import audit_stage,initial_confirmation_report
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise RuntimeError(f'Pipeline signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while read(INITIAL/'status.json')['phase']!='complete':
            parent_status=read(INITIAL/'status.json')
            if parent_status['phase']=='failed':raise RuntimeError(f'Initial 5K failed: {parent_status}')
            atomic(ROOT/'status.json',dict(phase='waiting_for_initial_5k',initial=parent_status,controller_pid=os.getpid()))
            time.sleep(10)
        atomic(ROOT/'status.json',dict(phase='auditing_initial_5k',controller_pid=os.getpid()))
        initial_confirmation_report()
        check=ROOT/'development_check.json'
        if not check.exists():
            atomic(ROOT/'status.json',dict(phase='fusion_development_check',controller_pid=os.getpid()))
            development_check()
        prepare_stage('tuning_1k',fusion.configurations());run_stage('tuning_1k')
        if read(ROOT/'tuning_1k/status.json')['phase']!='complete':return
        selection=choose_final()
        audited=[c['arm'] for c in selection['configs']]
        # Even if no fusion passes selection, audit both best fusion candidates
        # and the components they failed to exceed.
        audited+= [d[k]['arm'] for d in selection['decisions'] for k in ('best_fusion','best_component','best_native') if d[k] is not None]
        audit_stage(ROOT/'tuning_1k',sorted(set(audited)))
        if selection['configs']:
            prepare_stage('selected_5k',selection['configs'],selection);run_stage('selected_5k')
            if read(ROOT/'selected_5k/status.json')['phase']!='complete':return
            audit_stage(ROOT/'selected_5k',[c['arm'] for c in selection['configs']])
        atomic(ROOT/'status.json',dict(phase='complete',tuning_configurations=len(fusion.configurations()),
            confirmed_configurations=len(selection['configs']),selection=selection['decisions'],
            no_further_sampling_queued=True,research_goal_achieved=False))
    except BaseException as error:
        previous=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        atomic(ROOT/'status.json',dict(phase='failed',error=repr(error),previous=previous,controller_pid=os.getpid()))
        raise
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--pipeline',action='store_true');action.add_argument('--check',action='store_true')
    action.add_argument('--status',action='store_true');action.add_argument('--stop-after-current',action='store_true')
    action.add_argument('--worker',type=int,choices=range(4));parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--run-id');parser.add_argument('--parent-pid',type=int)
    args=parser.parse_args()
    if args.pipeline:pipeline()
    elif args.check:development_check()
    elif args.status:print(json.dumps(read(ROOT/'status.json'),ensure_ascii=False,indent=2))
    elif args.stop_after_current:ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'STOP_AFTER_CURRENT').touch()
    else:
        assert args.stage and args.run_id and args.parent_pid
        worker(args.stage,args.worker,args.run_id,args.parent_pid)
