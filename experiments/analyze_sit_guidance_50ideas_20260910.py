"""Post-run audit and readable rankings for the completed, frozen 649-arm study.

Does not change its request, sampler, samples, or frozen source files.
All 649 manifests are inspected; raw batches and independent cached-feature
metric arithmetic are rechecked for the explicitly listed headline arms.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from experiments.sit_guidance_portfolio_20260910 import runner as study
from experiments import run_sit_guidance_50ideas_20260910 as catalog
from experiments.analyze_official_sit_pfr_symmetric_20260909 import fid_components
from experiments.lifting_scale_sweep_20260909 import WORK, array_sha, atomic, read, sha

OUT=WORK/'docs/data/sit_guidance_portfolio_20260910'
REPORT=WORK/'docs/SIT_GUIDANCE_50_IDEAS_RESULTS_20260910_ZH.md'
CATALOG=WORK/'docs/SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md'
HYBRID={31,32,34,35}


def main():
    study.install_infrastructure();request,h=study.verify_request()
    status=read(study.ROOT/'status.json');results=read(study.ROOT/'results.json')
    assert status['phase']=='complete' and status['worker_exit_codes']==[0]*4
    assert len(results)==649 and {r['arm'] for r in results}==set(study.ARMS)
    assert all(r['complete'] for r in results)
    OUT.mkdir(parents=True,exist_ok=True)
    noise=np.load(study.BANK_ROOT/'noise.npy',mmap_mode='r');labels=np.load(study.BANK_ROOT/'labels.npy')
    assert array_sha(noise)==request['bank']['noise_sha256']
    assert array_sha(labels)==request['bank']['label_sha256']
    seen_count=0
    for row in results:
        assert row['request_sha256']==h and row['samples']==1000 and row['coverage_verified']
        assert row['noise_sha256']==request['bank']['noise_sha256']
        assert row['label_sha256']==request['bank']['label_sha256']
        assert all(row[k]==v for k,v in study.SPECS[row['arm']].items())
        directory=study.ROOT/row['arm']
        assert read(directory/'result.json')==row
        metrics=read(directory/'fid.json');assert metrics==row['metrics'] and metrics['sample_count']==1000
        assert metrics['fid']==row['fid'] and np.isfinite(row['fid'])
        seen=set()
        for rank in range(4):
            summary=read(directory/f'rank{rank}/summary.json')
            assert summary['complete'] and summary['request_sha256']==h
            assert summary['noise_sha256']==row['noise_sha256'] and summary['label_sha256']==row['label_sha256']
            for rec in summary['files']:
                start=rec['start'];assert start not in seen and (start//8)%4==rank
                assert rec['file']==f'batch{start:04d}.npz' and len(rec['sha256'])==64
                seen.add(start)
        assert seen==set(range(0,1000,8));seen_count+=len(seen)
    assert seen_count==81125
    checks=read(study.ROOT/'preflight_passed.json')
    assert checks['passed'] and checks['request_sha256']==h
    assert {a for c in checks['checks'] for a in c['grid_arms']}=={r['arm'] for r in results if r['role']=='candidate'}
    assert {r['idea'] for c in checks['checks'] for r in c['trajectories']}==set(range(1,51))
    def best(predicate):return min((r for r in results if predicate(r)),key=lambda r:r['fid'])
    ig=best(lambda r:r['role']=='control' and r['source']=='ig' and r['solver']=='heun64' and r['strength']>0)
    dopri=best(lambda r:r['role']=='control' and r['solver']=='dopri5')
    cfg=best(lambda r:r['role']=='control' and r['source']=='cfg')
    winners={i:best(lambda r:r['idea']==i) for i in range(1,51)}
    pure=best(lambda r:r['role']=='candidate' and r['source']=='ig' and r['idea'] not in HYBRID)
    headline={r['arm']:r for r in (ig,dopri,cfg,pure,winners[1],winners[38],winners[3])}
    references={}
    with np.load(study.infrastructure.REFERENCE) as ref:
        for key,mu,cov in [('pool_3','mu','sigma'),('spatial','mu_s','sigma_s')]:
            references[key]=(ref[mu].astype(float),ref[cov].astype(float))
    audited=[]
    for arm,row in headline.items():
        directory=study.ROOT/arm
        for name,key in [('samples.npz','sample_sha256'),('latents.npy','latents_sha256'),('activations.npz','activations_sha256')]:
            assert sha(directory/name)==row[key]
        total_seconds=0.;calls=np.zeros(3)
        for rank in range(4):
            base=directory/f'rank{rank}';summary=read(base/'summary.json')
            for rec in summary['files']:
                path=base/rec['file'];assert sha(path)==rec['sha256']
                start=rec['start']
                with np.load(path) as batch:
                    assert str(batch['request_sha256'])==h
                    assert str(batch['noise_sha256'])==array_sha(noise[start:start+8])
                    np.testing.assert_array_equal(batch['labels'],labels[start:start+8])
                    assert batch['latents'].shape==(8,4,32,32) and np.isfinite(batch['latents']).all()
                    assert batch['arr_0'].shape==(8,256,256,3) and batch['arr_0'].dtype==np.uint8
                    total_seconds+=float(batch['trajectory_seconds'])+float(batch['decode_seconds'])
                    calls+=np.array([int(batch[k]) for k in ('full_calls','prefix_calls','auxiliary_full_calls')])
        assert abs(total_seconds-row['sum_batch_gpu_seconds'])<1e-7
        for value,key in zip(calls/125,('full_calls_per_image','prefix_calls_per_image','auxiliary_full_calls_per_image')):
            assert value==row[key]
        key=dict(activations_sha256=row['activations_sha256'],reference_sha256=request['reference_sha256'],fid=row['fid'],sfid=row['metrics']['sfid'])
        path=OUT/f'audit_{arm}.json'
        if path.exists():
            check=read(path);assert check['key']==key
        else:
            values={}
            with np.load(directory/'activations.npz') as data:
                for name,(mean,cov) in references.items():
                    assert data[name].shape[0]==1000
                    values[name]=sum(fid_components(data[name].astype(float),mean,cov))
            check=dict(key=key,independent_fid=values['pool_3'],independent_sfid=values['spatial'],
                fid_absolute_error=abs(values['pool_3']-row['fid']),sfid_absolute_error=abs(values['spatial']-row['metrics']['sfid']))
            assert check['fid_absolute_error']<.001 and check['sfid_absolute_error']<.001
            atomic(path,check)
        audited.append(dict(arm=arm,raw_batches_and_aggregate_hashes_verified=True,costs_reconciled=True,**check))
        print(json.dumps(dict(audited=arm,fid=row['fid'])),flush=True)
    audit=dict(passed=True,request_sha256=h,all_649_metadata_manifests_passed=True,
        all_81125_batch_indices_covered=True,source_asset_bank_reference_hashes_verified=True,
        real_model_preflight_all_600_grid_points_and_50_trajectories=True,
        headline_raw_and_independent_metric_audits=audited,
        all_batch_bytes_rehashed_after_run=False,feature_extraction_repeated=False,
        bank_is_reused_exploratory_bank=True,independent_confirmation=False)
    atomic(OUT/'audit.json',audit)
    fields=['arm','idea','key','source','strength','theta','solver','cutoff','role','fid','sfid','inception_score',
            'full_calls_per_image','prefix_calls_per_image','auxiliary_full_calls_per_image','sum_batch_gpu_seconds']
    with (OUT/'all_649_results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in results:writer.writerow(dict(row,**{k:row['metrics'][k] for k in ('sfid','inception_score')}))
    def gain(row,baseline):return (1-row['fid']/baseline['fid'])*100
    def cost(row,baseline):return row['sum_batch_gpu_seconds']/baseline['sum_batch_gpu_seconds']
    lines=['# 50 个 guidance idea：全部 649 组小 SiT 1K 结果\n',
        f'全部 649/649 组完成（50×12 候选、49 对照），649000 张质量样本，耗时 {status["wall_seconds"]/3600:.2f} 小时；数值失败 0，四个 worker 正常退出。tmux `sit_guidance_50ideas_0910` 的采样 pane 已结束，退出码 0。\n',
        f'纯 IG 最好为 **#40 软空间局部化弱前缀：{pure["fid"]:.6f}**，同求解器已调强度 IG 为 {ig["fid"]:.6f}，改善 **{gain(pure,ig):.3f}%**；采样与解码成本 **{cost(pure,ig):.3f}×**。强度 .8、locality=2 均在各自网格内部。该候选 sFID 略差、IS 略好，尚非多指标一致改善。\n',
        f'全体最小 FID 为 **#1 CFG＋APG：{winners[1]["fid"]:.6f}**；原生 CFG 已达到 {cfg["fid"]:.6f}，所以该方法的相关增益为 **{gain(winners[1],cfg):.3f}%**，成本 **{cost(winners[1],cfg):.3f}×**。APG 是已有方法适配，且最佳 strength=2 位于本候选网格上界。不能把原生 CFG 相比 IG 的大幅优势计作新方法贡献。\n',
        '本次沿用历史 paired 1K noise/label bank，所有配置一致；没有新独立 5K。1K 的 600 点选择有乐观偏差，先把 #40 当作需复验的纯 IG 候选；不能称已证实优于 IG、已找到新颖论文方法，或将本表数值与旧 5K 相减。\n',
        '|配置|FID↓|sFID↓|IS↑|Full/prefix 每图|采样+解码 batch GPU秒|',
        '|---|---:|---:|---:|---|---:|']
    for title,row in [('最佳 Heun IG',ig),('最佳 Dopri5 IG',dopri),('最佳原生 CFG',cfg),
        ('#1 CFG＋APG',winners[1]),('#38 CFG 条件割线',winners[38]),('#3 CFG 通道方差回缩',winners[3]),('#40 纯 IG 局部注意力',pure)]:
        lines.append(f'|{title}|{row["fid"]:.6f}|{row["metrics"]["sfid"]:.5f}|{row["metrics"]["inception_score"]:.5f}|{row["full_calls_per_image"]:.3f}/{row["prefix_calls_per_image"]:.0f}|{row["sum_batch_gpu_seconds"]:.3f}|')
    lines.extend(['\n## 每个 idea 的最佳点\n',
        '41 个 source=IG 中，#31/#32/#34/#35 查询了 null 分支，列为混合方法；其余 37 项不需要 null。9 项为 CFG。表内所有 50 项均完成 12 组；Δ 为候选减对照，负数更好。成本相对同源基线；混合方法同时应对照 CFG。\n',
        '|ID|候选|来源|最佳FID|ΔIG|ΔCFG|强度/结构参数|边界点|成本倍数|',
        '|--:|---|---|--:|--:|--:|---|---|--:|'])
    for row in sorted(winners.values(),key=lambda r:r['fid']):
        idea=request['ideas'][row['idea']-1]
        strengths=catalog.IG_STRENGTHS if row['source']=='ig' else catalog.CFG_STRENGTHS
        bounds=[]
        if row['strength'] in (min(strengths),max(strengths)):bounds.append('strength')
        if row['theta'] in (min(idea['theta']),max(idea['theta'])):bounds.append(idea['theta_name'])
        source='IG+null' if row['idea'] in HYBRID else row['source'].upper()
        base=cfg if row['source']=='cfg' else ig
        lines.append(f'|{row["idea"]}|{idea["title"]}|{source}|{row["fid"]:.5f}|{row["fid"]-ig["fid"]:+.5f}|{row["fid"]-cfg["fid"]:+.5f}|{row["strength"]}/{row["theta"]}|{",".join(bounds) or "否"}|{cost(row,base):.3f}|')
    lines.extend(['\n## 可复核范围\n',
        '所有 649 个结果与冻结参数、输入身份、FID JSON 和全部 81125 个 batch 索引清单一致。原始采样器在每次评估前校验批次哈希并核对覆盖；本次额外重查表头七组的 875 个原始 batch、聚合样本/latent/feature 哈希和计时/NFE，并用缓存特征独立 FP64 重算 FID、sFID，误差均小于 .001。没有重新抽取 Inception 特征，也没有在这次报告中重读所有 81125 批图像字节。源码、模型/拟合、参考和输入哈希均通过。\n',
        f'冻结请求 SHA256：`{h}`。噪声：`{request["bank"]["noise_sha256"]}`。标签：`{request["bank"]["label_sha256"]}`。\n',
        '[逐组 CSV](data/sit_guidance_portfolio_20260910/all_649_results.csv) · [审计 JSON](data/sit_guidance_portfolio_20260910/audit.json) · [50 个候选依据及仓库差异](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。\n',
        f'原始输出目录：`{study.ROOT}`。目录中的 `launch.json` 记录实际 tmux 命令；冻结 catalog 为 `experiments/run_sit_guidance_50ideas_20260910.py`，实际调度入口是 `python -m experiments.sit_guidance_portfolio_20260910.runner --run-prepared`。原队列已完整结束，无需再次启动。\n'])
    REPORT.write_text('\n'.join(lines))
    lines=['# 已完成 50 idea 队列的候选说明\n',
        f'以下从冻结请求 `{h}` 导出，参数与实际完成的 600 组候选一致。这是运行后的可读说明，不把新文档冒充预先冻结的协议。每项 4 个 strength × 3 个结构参数；完整结果见 [结果报告](SIT_GUIDANCE_50_IDEAS_RESULTS_20260910_ZH.md)。\n',
        'IG strength=(.5,.65,.8,.95)，t<.25 乘6/7、.25≤t<.5 取峰值、其后0；CFG strength=(.5,1,1.5,2)，t<.75启用。所有候选使用64步Heun；历史只在接受步后更新，同一步两个stage共享扰动。原生IG另测12点Dopri5；CFG对照在cutoff=.5/.75各测12点。\n',
        'S/W 是条件强/弱速度，U 是 null Strong，D=S−W，Dc=S−U，b=1−t，m=z+bS，C 是既有浅层特征预测 gap；N_D 为恢复到原 gap 范数。第36项方向诊断基于替代类别gap；第23项Strang无普通RHS诊断。第49项原始对称线性误差场保守，cap后不保证保守。所有理论依据均是局部代数或建模假设，不是FID保证。\n']
    for idea in request['ideas']:
        lines.extend([f'## {idea["id"]:02d}. {idea["title"]}\n',
            f'键：`{idea["key"]}`；{idea["theta_name"]}={idea["theta"]}。\n',
            f'构造：`{idea["formula"]}`。\n',f'依据：{idea["rationale"]}\n',
            f'仓库近邻：[旧记录]({idea["prior"]})。变化：{idea["change"]}\n',
            f'限制：{idea["limitation"]}\n',
            '参考：'+', '.join(f'[{catalog.REFERENCES[k][0]}]({catalog.REFERENCES[k][1]})' for k in idea['references'])+'。\n'])
    CATALOG.write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(REPORT),all_649_complete=True,
        headline_audited=len(audited),best_pure_ig=pure['arm'],best_pure_ig_gain_percent=gain(pure,ig))),flush=True)


if __name__=='__main__':main()
