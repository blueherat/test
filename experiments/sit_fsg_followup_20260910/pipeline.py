"""Serial dependency queue using the existing audited, resumable shard engine."""
from __future__ import annotations
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from experiments.sit_fsg_followup_20260910 import core
from experiments import run_sit_fsg_20ideas_20260910 as catalog
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import EXPS,WORK,array_sha,atomic,read,sha

ROOT=EXPS/'sit_fsg_followup_20ideas_20260910'
WAIT_ROOT=EXPS/'sit_guidance_fusion_20260910'
STAGE='fsg_screen_1k'
MODULE='experiments.run_sit_fsg_20ideas_20260910'
PROTOCOL=WORK/'docs/SIT_FSG_20_IDEAS_PROTOCOL_20260910_ZH.md'
RESEARCH=WORK/'docs/FSG_FOLLOWUP_RESEARCH_20_IDEAS_20260910_ZH.md'
REPORT=WORK/'docs/SIT_FSG_20_IDEAS_RESULTS_20260910_ZH.md'


def sources():
    return [Path(__file__).resolve(),Path(core.__file__).resolve(),Path(catalog.__file__).resolve(),
        Path(engine.__file__).resolve(),Path(analysis.__file__).resolve(),
        WORK/'experiments/sit_guidance_fusion_20260910.py',PROTOCOL,RESEARCH]


def configure():
    # These are process-local bindings. Previously launched jobs and source
    # files are unaffected. All imported implementation files are hashed.
    engine.ROOT=ROOT;engine.MODULE=MODULE;engine.PROTOCOL=PROTOCOL
    engine.STAGES={STAGE:(1000,202610080)};engine.fusion=core
    engine.additional_sources=sources
    analysis.write_progress=write_progress


def write_progress(base,request,rows):
    valid=[r for r in rows if r['complete']]
    def best(filt):
        pool=[r for r in valid if filt(r)]
        return min(pool,key=lambda r:r['fid']) if pool else None
    native=best(lambda r:r['family']=='cfg_native')
    control=best(lambda r:r['source']=='cfg' and r['role']!='candidate')
    fields=['arm','idea_id','family','role','strength','theta','fid','sfid','inception_score',
        'full_calls_per_image','prefix_calls_per_image','auxiliary_full_calls_per_image','sum_batch_gpu_seconds','complete']
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=fields,extrasaction='ignore');writer.writeheader()
    for row in rows:
        value=dict(row)
        if row['complete']:value.update({k:row['metrics'][k] for k in ('sfid','inception_score')})
        writer.writerow(value)
    (base/'results.csv').write_text(output.getvalue())
    portable=WORK/'docs/data/sit_fsg_followup_20ideas_20260910';portable.mkdir(parents=True,exist_ok=True)
    (portable/'all_results.csv').write_text(output.getvalue())
    lines=['# FSG 后续二十项假设：1K 筛选\n',
        f'已完成 {len(rows)}/{len(request["configs"])} 组；其中数值失败 {sum(not r["complete"] for r in rows)}。每项10组，共200候选和56对照。\n',
        '所有方法使用同一套新噪声，每类10张。Δ=候选FID减对照，负值较好。当前排名来自调参，不是独立5K确认。\n']
    if native:lines.append(f'原生CFG当前最佳 **{native["fid"]:.6f}**，`{native["arm"]}`。\n')
    if control:lines.append(f'全部CFG对照当前最佳 **{control["fid"]:.6f}**，`{control["arm"]}`。APG和FSG算子适配只算对照。\n')
    lines.extend(['|ID|假设|完成/10|最佳FID|Δ原生CFG|Δ最佳对照|a / 参数|采样+解码成本×CFG|Full/prefix|',
        '|--:|---|--:|--:|--:|--:|---|--:|---|'])
    for idea in catalog.IDEAS:
        done=sum(r['idea_id']==idea['id'] for r in rows)
        row=best(lambda r:r['idea_id']==idea['id'])
        if row is None:lines.append(f'|{idea["id"]}|{idea["title"]}|{done}/10|—|—|—|—|—|—|');continue
        delta=lambda x:f'{row["fid"]-x["fid"]:+.6f}' if x else '—'
        cost=f'{row["sum_batch_gpu_seconds"]/native["sum_batch_gpu_seconds"]:.3f}' if native else '—'
        lines.append(f'|{idea["id"]}|{idea["title"]}|{done}/10|{row["fid"]:.6f}|{delta(native)}|{delta(control)}|{row["strength"]} / {row["theta"]}|{cost}|{row["full_calls_per_image"]:.0f}/{row["prefix_calls_per_image"]:.0f}|')
    lines.extend(['\n20项是明确改变目标或信息假设的研究候选，不声明20项均具有论文级原创性。局部残差、散度或体积改善不作为FID的替代。\n',
        f'原始结果：`{base}`。输入/实现请求SHA256：`{sha(base/"request.json")}`。\n',
        '[完整文献与各项区别](FSG_FOLLOWUP_RESEARCH_20_IDEAS_20260910_ZH.md) · [冻结协议](SIT_FSG_20_IDEAS_PROTOCOL_20260910_ZH.md) · [逐组CSV](data/sit_fsg_followup_20ideas_20260910/all_results.csv)。\n'])
    if (base/'analysis_audit.json').exists():
        audit=read(base/'analysis_audit.json')
        lines.append(f'\n元数据及全部覆盖核验完成；{len(audit["raw_and_metric_audits"])}个代表/最佳配置另外通过原始batch哈希、成本与缓存特征FP64 FID/sFID重算。没有重新抽取Inception特征。\n')
    text='\n'.join(lines);(base/'report.md').write_text(text);REPORT.write_text(text)


@torch.inference_mode()
def development_check():
    configure();cpu=core.cpu_checks();engine.verify_parent();ROOT.mkdir(parents=True,exist_ok=True)
    rt=engine.parent.operators.make_runtime()
    noise=torch.from_numpy(np.load(engine.parent.BANK_ROOT/'noise.npy')[:8].copy()).cuda()
    labels=torch.from_numpy(np.load(engine.parent.BANK_ROOT/'labels.npy')[:8].copy()).cuda()
    limits=core.limiting_checks(rt,noise,labels);hooks=engine.parent.hook_counts(rt)
    controls=catalog.configurations();strong=next(c for c in controls if c['family']=='strong')
    original,_=core.sample(rt,noise,labels,strong)
    # The conditioning intervention must cover the final AdaLN as well as all
    # transformer blocks: all and none are exact native branch endpoints.
    rt.labels=labels;ctx=core.old.StepContext(xi=torch.ones_like(noise))
    probe=core.Probe(rt,noise,.25,ctx)
    assert torch.equal(probe.layer_field('all'),probe.c)
    null=probe.layer_field('none')
    assert torch.equal(null,probe.u),float((null-probe.u).abs().max())
    # Capture the original time embedding and construct t_embed+eu directly;
    # subtracting ec from rounded t_embed+ec would not reproduce native null.
    intervention_error=float((null-probe.u).abs().max())
    fields=[]
    rt.labels=labels[:2]
    generator=torch.Generator(device='cuda').manual_seed(202610079)
    for cfg in controls:
        if cfg['role']!='candidate':continue
        n=noise[:2];ctx=core.old.StepContext(xi=torch.randn(n.shape,device=n.device,generator=generator))
        ctx.block_permutation=torch.argsort(torch.rand(2,64,device=n.device,generator=generator),dim=1)
        if cfg['key'] in core.FIELD_KEYS:
            got,info=core.evaluate(rt,n,.25,cfg,cfg['strength'],ctx)
            assert torch.isfinite(got).all() and torch.isfinite(info['diagnostics']).all(),cfg['arm']
        else:
            got,rec=core.calibrate(rt,n,.25,cfg,ctx,cfg['strength'])
            assert torch.isfinite(got).all() and all(np.isfinite(v) for v in rec.values()),cfg['arm']
        assert engine.parent.hook_counts(rt)==hooks and rt.labels is not None
        fields.append(cfg['arm'])
        if len(fields)%20==0:print(json.dumps(dict(grid_probe_completed=len(fields),total=200)),flush=True)
    trajectories=[]
    for row in catalog.IDEAS:
        cfg=[c for c in controls if c['key']==row['key']][-1]
        latent,stats=core.sample(rt,noise,labels,cfg)
        assert torch.isfinite(latent).all() and np.isfinite(stats['diagnostics']).all(),cfg['arm']
        zero,_=core.sample(rt,noise,labels,cfg,zero=True)
        assert torch.equal(zero,original),cfg['key']
        assert engine.parent.hook_counts(rt)==hooks and rt.labels is labels
        trajectories.append(dict(arm=cfg['arm'],full=stats['full_calls'],prefix=stats['prefix_calls'],
            auxiliary=stats['auxiliary_full_calls'],max_abs=float(latent.abs().max()),
            accepted=stats.get('accepted_calibrations'),shift=stats.get('calibration_shift_norm_sum')))
        print(json.dumps(dict(trajectory_check=row['id'],**trajectories[-1])),flush=True)
    # Include both newly implemented calibration controls in real trajectories.
    for family in ('fsg_operator_control','plain_root_control'):
        cfg=[c for c in controls if c['family']==family][-1]
        latent,stats=core.sample(rt,noise,labels,cfg);assert torch.isfinite(latent).all()
        trajectories.append(dict(arm=cfg['arm'],full=stats['full_calls'],control=True))
    golden=[]
    for family in ('ig_native','ig_dopri','ig_local','cfg_native','cfg_apg'):
        pairs=[(c,r) for c,r in engine.golden_pairs(controls) if c['family']==family]
        assert pairs,family
        c,r=pairs[0];latent,_=core.sample(rt,noise,labels,c)
        path=engine.parent.ROOT/r['arm']/'rank0/batch0000.npz'
        with np.load(path) as batch:np.testing.assert_array_equal(latent.cpu().numpy(),batch['latents'])
        golden.append(dict(family=family,old_arm=r['arm'],sha256=sha(path),exact=True))
    repeat,_=core.sample(rt,noise,labels,strong);assert torch.equal(repeat,original)
    result=dict(passed=True,cpu=cpu,limits=limits,grid_probes=fields,trajectories=trajectories,
        golden=golden,intervention_null_max_abs_error=intervention_error,
        zero_trajectories_exact=True,native_after_interventions_exact=True,
        source_hashes={str(p):sha(p) for p in sources()},runtime_sources=rt.sources,no_fid_used=True)
    atomic(ROOT/'development_check.json',result)
    print(json.dumps(dict(development_check_passed=True,ideas=20,grid=200,trajectories=len(trajectories))),flush=True)


def pipeline():
    configure();ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise RuntimeError(f'Pipeline signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        check=read(ROOT/'development_check.json');assert check['passed']
        for path,digest in check['source_hashes'].items():assert sha(path)==digest,path
        while True:
            dependency=read(WAIT_ROOT/'status.json')
            if dependency['phase']=='complete':break
            if dependency['phase']=='failed':raise RuntimeError(f'Fusion dependency failed: {dependency}')
            atomic(ROOT/'status.json',dict(phase='waiting_for_fusion',controller_pid=os.getpid(),
                dependency_root=str(WAIT_ROOT),dependency=dependency,ideas=20,candidate_arms=200,total_arms=256))
            time.sleep(10)
        engine.prepare_stage(STAGE,catalog.configurations());engine.run_stage(STAGE)
        if read(ROOT/STAGE/'status.json')['phase']!='complete':return
        rows=read(ROOT/STAGE/'results.json')
        selected=[]
        for family in dict.fromkeys(c['family'] for c in catalog.configurations()):
            valid=[r for r in rows if r['family']==family and r['complete']]
            if valid:selected.append(min(valid,key=lambda r:r['fid'])['arm'])
        analysis.audit_stage(ROOT/STAGE,selected)
        write_progress(ROOT/STAGE,read(ROOT/STAGE/'request.json'),rows)
        atomic(ROOT/'status.json',dict(phase='complete',candidate_arms=200,control_arms=56,
            total_arms=256,samples_per_arm=1000,numerical_failures=sum(not r['complete'] for r in rows),
            audited_best_per_family=selected,no_further_sampling_queued=True,research_goal_achieved=False))
    except BaseException as error:
        previous=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        atomic(ROOT/'status.json',dict(phase='failed',error=repr(error),previous=previous,controller_pid=os.getpid()))
        raise
    finally:lock.close()


def main():
    configure();parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--pipeline',action='store_true');action.add_argument('--check',action='store_true')
    action.add_argument('--status',action='store_true');action.add_argument('--stop-after-current',action='store_true')
    action.add_argument('--worker',type=int,choices=range(4))
    parser.add_argument('--stage',choices=[STAGE]);parser.add_argument('--run-id');parser.add_argument('--parent-pid',type=int)
    args=parser.parse_args()
    if args.pipeline:pipeline()
    elif args.check:development_check()
    elif args.status:print(json.dumps(read(ROOT/'status.json'),ensure_ascii=False,indent=2))
    elif args.stop_after_current:ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'STOP_AFTER_CURRENT').touch()
    else:
        assert args.stage and args.run_id and args.parent_pid
        engine.worker(args.stage,args.worker,args.run_id,args.parent_pid)
