"""Detached real-data SSG capacity comparison; exact eight-point 5K refinements."""
import json
from pathlib import Path
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.weak_reference_loss_20260914.idle import gpu_snapshot, eligible
from .jit_ssg import ROOT

lock=threading.Lock()
claimed=set()


class Device:
    def __enter__(self):
        while True:
            if (ROOT/'STOP_AFTER_CURRENT').exists():raise RuntimeError('Pipeline stop marker')
            with lock:
                rows=gpu_snapshot()
                row=next((r for r in rows if r['index'] in (1,2,3) and r['index'] not in claimed and eligible(r)),None)
                if row is not None:
                    self.index=row['index'];claimed.add(self.index);return self.index
            time.sleep(5)

    def __exit__(self,*unused):
        with lock:claimed.remove(self.index)


def launch(task,output,arguments):
    if (output/'complete.json').exists():return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Unfinished existing output requires review: {output}')
    with Device() as gpu:
        cmd=[c.PYTHON,'-u','-m','classifier_guidance.launch','--gpus',str(gpu),
             '--output',str(output),'--task',task,'--',*arguments]
        print(json.dumps(dict(event='launch',gpu=gpu,output=str(output),command=cmd)),flush=True)
        subprocess.run(cmd,cwd=c.WORK,check=True)
        if not (output/'complete.json').exists():raise RuntimeError(f'Incomplete: {output}')


def train(blocks):
    output=ROOT/f'blocks{blocks}/training_50k'
    checkpoint=ROOT/f'preflight_fullbatch/blocks{blocks}/checkpoint_000100.pt'
    launch('jit-ssg',output,['--blocks',str(blocks),'--steps','50000','--microbatch','256','--resume',str(checkpoint)])
    return blocks


def point(blocks,w,cfg=1.,mode='guided'):
    key=f'{mode}_cfg{cfg:g}_w{w:.2f}'
    output=ROOT/f'blocks{blocks}/points'/key
    args=['--blocks',str(blocks),'--w',str(w),'--cfg',str(cfg),'--mode',mode]
    if mode!='baseline':args+=['--checkpoint',str(ROOT/f'blocks{blocks}/training_50k/checkpoint_050000.pt')]
    launch('jit-ssg-eval',output,args)
    metric=c.read(output/'metrics.json')
    assert metric['n']==5000
    return dict(w=w,extra_a=w-1,cfg=cfg,mode=mode,fid=metric['fid'],inception_score=metric['inception_score'],metrics=str(output/'metrics.json'))


def scan(blocks):
    arm=ROOT/f'blocks{blocks}';results={}
    # All heads share the no-guidance baseline and the full [0,1] SSG interval.
    for tick in (120,140,160,180):results[tick]=point(blocks,tick/100)
    best=min(results,key=lambda t:(results[t]['fid'],t))
    low=max(105,min(best-15,160))
    fine=list(range(low,low+36,5))
    assert len(fine)==8 and fine[-1]<200
    c.atomic(arm/'fine_plan.json',dict(coarse_best_w=best/100,w=[v/100 for v in fine],
        count=8,spacing=.05,n_per_point=5000,coarse_reused=True,no_independent_validation=True,
        upper_boundary_not_extended_to_2=True))
    for tick in fine:
        if tick not in results:results[tick]=point(blocks,tick/100)
    weak=point(blocks,1.,mode='weak')
    fixed_cfg=point(blocks,1.2,cfg=3.)
    best=min(results,key=lambda t:(results[t]['fid'],t))
    result=dict(complete=True,blocks=blocks,points=[results[t] for t in sorted(results)],best=results[best],
        weak_only=weak,fixed_cfg3_w1_2=fixed_cfg,common_w1_6=results[160],
        best_at_upper_boundary=best==max(results),primary='no CFG, full-trajectory SSG')
    c.atomic(arm/'results.json',result);return result


def main():
    # Required preceding SiT tests and JiT implementation checks are gates.
    sit=ROOT.parent/'sit_mlp_capacity_diffusion_20260920/head_strength_audit/results.json'
    assert c.read(sit)['complete']
    assert c.read(ROOT/'paired_training_inputs.json')['passed']
    assert c.read(ROOT/'resume_audit/result.json')['passed']
    for p in (ROOT/'sampler_audits/blocks1/parity.json',ROOT/'sampler_audits/blocks2_cfg/parity.json'):
        assert c.read(p)['published_ssg_sampling_endpoint_exact']
    plan=dict(created_utc=c.now(),data='only full real ImageNet training set; no synthetic data',
        models='frozen JiT-B/16 EMA1; layer6; linear / 1 block / 2 blocks',
        train_steps=50000,global_batch=256,lr=5e-5,betas=[.9,.95],ema=.9999,
        training_order=[2,1,0],gpus='idle only from 1,2,3; preserve other compute processes',
        primary_sweep=dict(w_coarse=[1.2,1.4,1.6,1.8],fine_points=8,fine_spacing=.05,n_each=5000,
            shared_w1_baseline=True,no_w2=True,no_independent_validation=True),
        extra_controls='weak-only 5K; common w1.6 reuses coarse; CFG3 with w1.2; baseline CFG3',
        parameterization='paper total w; extra a=w-1',training_budget_is_not_exact_paper_epochs=True)
    c.atomic(ROOT/'plan.json',plan)
    c.atomic(ROOT/'status.json',dict(phase='training',updated_utc=c.now()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for done in as_completed([pool.submit(train,b) for b in (2,1,0)]):
            print('finished 50K',done.result(),flush=True)
    c.atomic(ROOT/'status.json',dict(phase='baselines',updated_utc=c.now()))
    baselines={str(cfg):point(0,1.,cfg=cfg,mode='baseline') for cfg in (1.,3.)}
    c.atomic(ROOT/'baselines.json',baselines)
    c.atomic(ROOT/'status.json',dict(phase='5k_sweeps',updated_utc=c.now()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=[done.result() for done in as_completed([pool.submit(scan,b) for b in (0,1,2)])]
    c.atomic(ROOT/'results.json',dict(complete=True,baselines=baselines,arms=sorted(results,key=lambda r:r['blocks'])))
    c.atomic(ROOT/'status.json',dict(phase='complete',updated_utc=c.now()))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        c.atomic(ROOT/'status.json',dict(phase='error',error=repr(exc),updated_utc=c.now()))
        raise
