"""Paired image experiments for specific FSG and CFG-Ctrl mechanisms."""
from __future__ import annotations
import argparse
import csv
import io
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
from experiments.sit_fsg_ctrl_hypothesis_20260911 import core
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as old_analysis
from experiments.sit_fsg_pasted_20260910 import core as semantic
from experiments.lifting_scale_sweep_20260909 import EXPS, WORK, atomic, read, sha, array_sha

ROOT = EXPS/'sit_fsg_ctrl_hypothesis_20260911'
SOURCE = EXPS/'sit_control_output_50ideas_20260910/control_screen_1k'
STAGE = 'hypothesis_1k'
MODULE = 'experiments.sit_fsg_ctrl_hypothesis_20260911.pipeline'
PROTOCOL = WORK/'docs/SIT_FSG_CTRL_HYPOTHESIS_PROTOCOL_20260911_ZH.md'
PORTABLE = WORK/'docs/data/sit_fsg_ctrl_hypothesis_20260911'
REUSED = dict(cfg_tuned='cfg_native_04', cfg_high='cfg_native_10',
              fsg_high='fsg_operator_control_07', smc_high='cfg_smc_control_07')
HANDOFF_METHODS = ('cfg_tuned','cfg_high','fsg_high','smc_high','instant_high')
_evaluate = engine.infrastructure.evaluate


def configurations():
    rows = []
    def add(method, handoff=None, tail='null'):
        values = core.METHODS[method]
        arm = method if handoff is None else f'{method}_{tail}_{handoff:02d}'
        rows.append(dict(arm=arm, family=method, key='mechanism', source='cfg',
            strength=values['amount'], theta=values.get('gain',0.), solver='heun64',
            cutoff=.75, role='mechanism_control', idea_id=None, external_semantics=False,
            parameters=dict(method=method,handoff=handoff,tail=tail)))
    for method in core.METHODS:
        if method not in REUSED:
            add(method)
    for method in HANDOFF_METHODS:
        for step in (16,32,48):
            add(method,step)
        add(method,32,'conditional')
    return rows


def sources():
    return [*sorted(Path(__file__).parent.glob('*.py')), PROTOCOL,
            Path(semantic.__file__).resolve(), Path(core.legacy_fsg.__file__).resolve()]


def verify():
    request = read(ROOT/'study_request.json')
    for group in ('sources','assets','references'):
        for path,digest in request[group].items():
            assert sha(path)==digest,path
    for name,digest in request['bank_files'].items():
        assert sha(ROOT/'inputs'/name)==digest,name
    return request


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'study_request.json').exists():
        return verify()
    check=read(ROOT/'development_check.json')
    assert check['passed']
    for path,digest in check['source_hashes'].items():
        assert sha(path)==digest,path
    parent=read(SOURCE/'request.json')
    for group in ('sources','assets'):
        for path,digest in parent[group].items():
            assert sha(path)==digest,path
    for name,digest in parent['bank_files'].items():
        assert sha(SOURCE/'inputs'/name)==digest,name
    bank=ROOT/'inputs';bank.mkdir()
    generator=torch.Generator(device='cuda').manual_seed(202612111)
    noise=torch.cat([torch.randn((8,4,32,32),device='cuda',generator=generator)
                     for _ in range(25)]).cpu().numpy()
    labels=np.random.default_rng(202612112).permutation(np.repeat(np.arange(100,dtype=np.int64),2))
    np.save(bank/'noise.npy',noise);np.save(bank/'labels.npy',labels)
    sources_map={**parent['sources'],**{str(p.resolve()):sha(p) for p in sources()}}
    assets={**parent['assets'],**{str(p):sha(p) for p in semantic.source_assets()}}
    references={str(p):sha(p) for p in [SOURCE/'request.json',ROOT/'development_check.json']}
    for arm in REUSED.values():
        commit=read(SOURCE/arm/'commit.json')
        assert commit['request_sha256']==sha(SOURCE/'request.json')
        for name,digest in commit['files'].items():
            assert sha(SOURCE/arm/name)==digest,name
        for name in ('commit.json','result.json'):
            references[str(SOURCE/arm/name)]=sha(SOURCE/arm/name)
    ident=dict(samples=200,intervention_samples=64,jacobian_samples=32,batch=8,ranks=4,
        noise_seed=202612111,label_seed=202612112,noise_sha256=array_sha(noise),
        label_sha256=array_sha(labels),time_steps=[8,16,24,32,40,48],
        intervention_steps=[8,24,40],jacobian_horizons=[2/64,4/64,8/64,16/64],
        sources=sources_map,assets=assets,references=references,
        bank_files={name:sha(bank/name) for name in ('noise.npy','labels.npy')},
        classifiers_optimized=False,created_unix=time.time())
    assert array_sha(noise)!=parent['bank']['noise_sha256']
    atomic(ROOT/'study_request.json',ident)
    base=ROOT/STAGE;base.mkdir();(base/'inputs').mkdir()
    for name in ('noise.npy','labels.npy'):
        shutil.copyfile(SOURCE/'inputs'/name,base/'inputs'/name)
        assert sha(base/'inputs'/name)==parent['bank_files'][name]
    configs=configurations()
    request=dict(parent,stage=STAGE,configs=configs,arms=[c['arm'] for c in configs],
        sources=sources_map,assets=assets,references=references,bank_root=str(base/'inputs'),
        independent_confirmation=False,selection=None,paired_existing_bank=True,
        reused_controls=REUSED,estimated_output_bytes=len(configs)*1000*450000)
    atomic(base/'request.json',request)
    snapshot=ROOT/'source_snapshot';snapshot.mkdir()
    for i,path in enumerate(sorted(sources_map)):
        shutil.copyfile(path,snapshot/f'{i:03d}_{Path(path).name}')
    atomic(ROOT/'status.json',dict(phase='prepared',quality_arms=len(configs),study_samples=200))
    return verify()


def progress(base,request,rows):
    PORTABLE.mkdir(parents=True,exist_ok=True)
    fields=['arm','fid','full_calls_per_image','prefix_calls_per_image',
            'sum_batch_gpu_seconds','complete']
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(rows)
    (PORTABLE/'quality_progress.csv').write_text(stream.getvalue())


def configure():
    engine.ROOT=ROOT;engine.MODULE=MODULE;engine.PROTOCOL=PROTOCOL
    engine.STAGES={STAGE:(1000,202610100)};engine.fusion=core
    old_analysis.write_progress=progress


def launch_study(part):
    verify()
    folder=ROOT/part;folder.mkdir(exist_ok=True)
    streams=[];processes=[]
    try:
        for rank in range(4):
            path=folder/f'rank{rank}.json'
            if path.exists() and read(path).get('complete'):
                continue
            stream=(folder/f'rank{rank}.log').open('a');streams.append(stream)
            processes.append(subprocess.Popen([engine.PYTHON,'-u','-m',MODULE,
                '--study-worker',part,'--rank',str(rank)],cwd=WORK,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
        atomic(ROOT/'status.json',dict(phase=part,worker_pids=[p.pid for p in processes],
            controller_pid=os.getpid(),started_unix=time.time()))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError(f'{part} worker failed; inspect rank logs')
            time.sleep(2)
        assert all(p.returncode==0 for p in processes)
        assert all(read(folder/f'rank{r}.json')['complete'] for r in range(4))
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=30)
        for s in streams:s.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true');parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--quality',action='store_true')
    parser.add_argument('--study',choices=['trajectories','interventions','jacobian'])
    parser.add_argument('--study-worker',choices=['trajectories','interventions','jacobian'])
    parser.add_argument('--rank',type=int)
    parser.add_argument('--worker',type=int);parser.add_argument('--stage',default=STAGE)
    parser.add_argument('--run-id');parser.add_argument('--parent-pid',type=int)
    args=parser.parse_args();configure()
    if args.check:
        from . import checks
        checks.run()
    elif args.prepare:prepare()
    elif args.worker is not None:
        engine.worker(args.stage,args.worker,args.run_id,args.parent_pid)
    elif args.study_worker:
        from . import study
        study.run_rank(args.study_worker,args.rank)
    elif args.study:launch_study(args.study)
    elif args.quality:engine.run_stage(STAGE)
    elif args.run:
        prepare()
        for part in ('trajectories','jacobian','interventions'):
            launch_study(part)
        engine.run_stage(STAGE)
        atomic(ROOT/'status.json',dict(phase='complete',study_complete=True,quality_complete=True,
            no_further_sampling_queued=True,finished_unix=time.time()))


if __name__=='__main__':main()
