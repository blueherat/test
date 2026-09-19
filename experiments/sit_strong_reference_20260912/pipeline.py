from __future__ import annotations
import argparse
import csv
import fcntl
import io
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import catalog, core, train, checks
from .catalog import ROOT, WORK, OLD, STAGE, PROTOCOL
from experiments import sit_guidance_fusion_pipeline_20260910 as engine
from experiments import analyze_sit_guidance_followup_20260910 as analysis
from experiments.lifting_scale_sweep_20260909 import atomic, read, sha

MODULE = 'experiments.sit_strong_reference_20260912.pipeline'
REPORT = WORK / 'docs/SIT_STRONG_REFERENCE_RESULTS_20260912_ZH.md'
PARENT_VERIFY = engine.verify_parent


def sources():
    return sorted(set([*Path(__file__).resolve().parent.glob('*.py'), PROTOCOL,
        Path(engine.__file__), Path(analysis.__file__), Path(core.previous.__file__),
        Path(core.previous.angular.__file__), *train.training_sources()]))


def verify_parent():
    parent = PARENT_VERIFY(); train.verify()
    assets = dict(parent['assets'])
    assets[str(ROOT/'training_request.json')] = sha(ROOT/'training_request.json')
    receipts = {}
    for method in (*catalog.METHODS,*catalog.REUSED_METHODS):
        folder = catalog.head_folder(method); done = read(folder/'complete.json')
        assert done['passed'] and done['step'] == catalog.STEPS and done['strong_unchanged']
        expected_request=(ROOT if method in catalog.METHODS else catalog.PARENT_ROOT)/'training_request.json'
        assert done['request_sha256'] == sha(expected_request)
        assert done['checkpoint_sha256'] == sha(folder/'model.pt')
        for name in ('complete.json','model.pt','input_fingerprints.json'):
            assets[str(folder/name)] = sha(folder/name)
        receipts[method] = done
    for source in ('ig','cfg'):
        assert len({v['input_fingerprints_sha256'] for k,v in receipts.items() if k.startswith(source+'_')})==1
    assert receipts['ig_strong']['paired_fingerprints_sha256']==receipts['ig_teacherstrong']['paired_fingerprints_sha256']
    for method in catalog.METHODS:
        path=catalog.head_folder(method)/'paired_fingerprints.json';assets[str(path)]=sha(path)
    return dict(parent, assets=assets)


def configure():
    engine.ROOT, engine.MODULE, engine.PROTOCOL = ROOT, MODULE, PROTOCOL
    engine.STAGES = {STAGE: (1000, catalog.NOISE_SEED)}
    engine.fusion, engine.additional_sources, engine.verify_parent = core, sources, verify_parent
    analysis.write_progress = write_progress


def write_progress(base, request, rows):
    stream = io.StringIO()
    fields = ['arm','source','role','strength','fid','full_calls_per_image','prefix_calls_per_image','sum_batch_gpu_seconds','complete']
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    (base/'results.csv').write_text(stream.getvalue())
    portable = WORK/'docs/data/sit_strong_reference_20260912'; portable.mkdir(parents=True,exist_ok=True)
    (portable/'all_results.csv').write_text(stream.getvalue())
    lines = ['# 冻结strong：预测蒸馏与生成分布参考\n',
        f'已完成 {len(rows)}/{len(request["configs"])} 组配对1K。强主干冻结，IG/CFG均不增加主干查询。\n',
        '|方法|FID↓|sFID↓|IS↑|Full/prefix|采样及解码GPU秒|', '|---|--:|--:|--:|--:|--:|']
    for row in rows:
        if not row['complete']:
            lines.append(f'|{row["arm"]}|数值失败|—|—|—|—|'); continue
        lines.append(f'|{row["arm"]}|{row["fid"]:.6f}|{row["metrics"]["sfid"]:.6f}|{row["metrics"]["inception_score"]:.4f}|'
            f'{row["full_calls_per_image"]:.0f}/{row["prefix_calls_per_image"]:.0f}|{row["sum_batch_gpu_seconds"]:.2f}|')
    lines += ['\nIG 比较 real / strong-generated 与原 FM / teacher-prediction 两种目标。CFG strong-data 使用原 FM 目标。已有 real / guided-data 头复用前轮权重；所有质量组重新生成。四个新头各1500步，初始化及噪声时间索引随机流配对。\n',
        '|训练|读出参数|训练GPU秒|验证guided MSE前→后|', '|---|--:|--:|--:|']
    for method in catalog.METHODS:
        done = read(ROOT/'training'/method/'complete.json')
        lines.append(f'|{method}|{done["trainable_parameters"]}|{done["training_seconds"]:.2f}|'
            f'{done["validation_before"]["guided_mse"]:.6f}→{done["validation_after"]["guided_mse"]:.6f}|')
    lines += ['\n预测风险改善不自动等于生成质量改善。此1K沿用历史探索bank，不是独立确认；未调guidance强度。\n',
        '[冻结协议与理论边界](SIT_STRONG_REFERENCE_PROTOCOL_20260912_ZH.md) · [逐组数据](data/sit_strong_reference_20260912/all_results.csv)\n',
        f'请求SHA256：{sha(base/"request.json")}。\n']
    output='\n'.join(lines); (base/'report.md').write_text(output); REPORT.write_text(output)


@torch.inference_mode()
def check_rank(rank):
    configure(); verify_parent(); core.install()
    rt = engine.parent.operators.make_runtime()
    n = torch.from_numpy(np.load(OLD/'inputs/noise.npy')[:8].copy()).cuda()
    y = torch.from_numpy(np.load(OLD/'inputs/labels.npy')[:8].copy()).cuda()
    rt.labels=y; hooks=engine.parent.hook_counts(rt); original_head=rt.head; original_final=rt.model.final_layer
    configs=catalog.configurations(); anchor=next(c for c in configs if c['family']=='strong')
    native,_=core.sample(rt,n[:2],y[:2],anchor); records=[]
    # Newly copied readouts reproduce the original velocity convention, including
    # the strong head's unused variance channels.
    for source in ('ig','cfg'):
        ts=n.new_full((2,),.25)
        f,context,strong=train.features_and_strong(rt,n[:2],ts,y[:2],source)
        rt.labels=y[:2]
        torch.testing.assert_close(strong,rt.field(n[:2],n.new_tensor(.25),'full'),rtol=0,atol=0)
        copied=train.initial_head(rt,source)
        value=train.project(rt,copied,f,context)
        rt.labels=y[:2] if source=='ig' else torch.full_like(y[:2],100)
        reference=rt.field(n[:2],n.new_tensor(.25),'base' if source=='ig' else 'full')
        torch.testing.assert_close(value,reference,rtol=0,atol=0)
        rt.labels=y
    for config in configs[rank::4]:
        z,stats=core.sample(rt,n[:2],y[:2],config)
        zero,_=core.sample(rt,n[:2],y[:2],config,zero=True)
        assert torch.isfinite(z).all() and torch.equal(zero,native),config['arm']
        assert engine.parent.hook_counts(rt)==hooks and rt.head is original_head and rt.model.final_layer is original_final
        records.append(dict(arm=config['arm'],max_abs=float(z.abs().max()),full_calls=stats['full_calls'],
            prefix_calls=stats['prefix_calls'],zero_exact=True))
    golden=next(c for c in configs if c['arm']==('strong_00','ig_local_00','cfg_native_04','cfg_apg_07')[rank])
    z,_=core.sample(rt,n,y,golden)
    with np.load(OLD/golden['arm']/'rank0/batch0000.npz') as saved:
        np.testing.assert_array_equal(z.cpu().numpy(),saved['latents'])
    for source in ('ig','cfg'):
        ordinary=next(c for c in configs if c['arm']==source+'_original_00')
        z,_=core.sample(rt,n,y,ordinary)
        path=(catalog.MEASURE/'measure_screen_1k/ig_native_matched_00/rank0/batch0000.npz' if source=='ig'
            else OLD/'cfg_native_04/rank0/batch0000.npz')
        with np.load(path) as saved:
            np.testing.assert_array_equal(z.cpu().numpy(),saved['latents'])
    again,_=core.sample(rt,n[:2],y[:2],anchor)
    assert torch.equal(again,native) and rt.head is original_head and rt.model.final_layer is original_final
    atomic(ROOT/'development'/f'rank{rank}.json',dict(passed=True,rank=rank,records=records,
        old_baseline_exact=True,ordinary_ig_cfg_trajectories_exact=True,strong_unchanged=True,
        source_hashes={str(p):sha(p) for p in sources()},runtime_sources=rt.sources))


def development_check():
    configure(); verify_parent(); folder=ROOT/'development'; folder.mkdir(exist_ok=True)
    analytic=checks.analytic_checks(); atomic(ROOT/'analytic_checks.json',analytic)
    processes=[];streams=[]
    try:
        for rank in range(4):
            log=(folder/f'rank{rank}.log').open('a');streams.append(log)
            processes.append(subprocess.Popen([engine.PYTHON,'-u','-m',MODULE,'--check-rank',str(rank)],
                cwd=WORK,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError(f'Preflight failed: {[p.poll() for p in processes]}')
            time.sleep(2)
        assert all(p.returncode==0 for p in processes)
        ranks=[read(folder/f'rank{r}.json') for r in range(4)]
        records=[record for rank in ranks for record in rank['records']]
        assert sorted(r['arm'] for r in records)==sorted(c['arm'] for c in catalog.configurations())
        expected={str(p):sha(p) for p in sources()}
        assert all(r['passed'] and r['source_hashes']==expected for r in ranks)
        atomic(ROOT/'development_check.json',dict(passed=True,trajectories=len(records),
            source_hashes=expected,all_zero_strength_exact=True,four_original_baselines_exact=True,
            ordinary_ig_cfg_trajectories_exact=True,analytic=analytic,no_fid_used=True))
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=30)
        for log in streams:log.close()


def pipeline():
    configure(); ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'STOP_AFTER_CURRENT').exists():return
        verify_parent()
        if not (ROOT/'development_check.json').exists():development_check()
        engine.prepare_stage(STAGE,catalog.planned())
        bank=read(ROOT/STAGE/'request.json')['bank']; old=read(OLD/'request.json')['bank']
        assert bank['noise_sha256']==old['noise_sha256'] and bank['label_sha256']==old['label_sha256']
        engine.run_stage(STAGE)
        if read(ROOT/STAGE/'status.json')['phase']!='complete':return
        rows=read(ROOT/STAGE/'results.json')
        analysis.audit_stage(ROOT/STAGE,[r['arm'] for r in rows if r['complete']])
        write_progress(ROOT/STAGE,read(ROOT/STAGE/'request.json'),rows)
        by_arm={r['arm']:r for r in rows}
        eligible=[]
        comparison={
            'ig': dict(candidates=('ig_teacher_00','ig_strong_00','ig_teacherstrong_00'),
                controls=('ig_original_00','ig_real_00','ig_ig_00','ig_half_00','ig_adg_reference_00')),
            'cfg': dict(candidates=('cfg_strong_00',),
                controls=('cfg_original_00','cfg_real_00','cfg_cfg_00','cfg_apg_07'))}
        for source,spec in comparison.items():
            controls=[by_arm[a] for a in spec['controls']]
            qualifying=[by_arm[a] for a in spec['candidates'] if by_arm[a]['complete']
                and all(r['complete'] and by_arm[a]['fid']<=r['fid']-.5 for r in controls)]
            if qualifying:eligible.append(min(qualifying,key=lambda r:r['fid'])['arm'])
        atomic(ROOT/'screen_review.json',dict(eligible_for_fixed_5k=eligible,threshold=.5,
            request_sha256=sha(ROOT/STAGE/'request.json'),results_sha256=sha(ROOT/STAGE/'results.json'),
            no_parameter_expansion=True))
        atomic(ROOT/'status.json',dict(phase='complete',total_arms=len(rows),eligible_for_fixed_5k=eligible,
            numerical_failures=sum(not r['complete'] for r in rows),final_audit_passed=True))


if __name__=='__main__':
    configure(); parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pipeline',action='store_true');group.add_argument('--check',action='store_true')
    group.add_argument('--check-rank',type=int);group.add_argument('--worker',type=int)
    parser.add_argument('--stage');parser.add_argument('--run-id');parser.add_argument('--parent-pid',type=int)
    args=parser.parse_args()
    if args.pipeline:pipeline()
    elif args.check:development_check()
    elif args.check_rank is not None:check_rank(args.check_rank)
    else:engine.worker(args.stage,args.worker,args.run_id,args.parent_pid)
