"""Read all planned strength points, audit FID arithmetic, and draw full curves."""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.lifting_scale_sweep_20260909 import ROOT, WORK, SMALL_DATA, MODELS, read, sha, atomic
from experiments.analyze_official_sit_pfr_symmetric_20260909 import REF, fid_components

OUT=WORK/'docs/data/lifting_wide_scale_20260909'


def render_report(request, report, rows):
    """Describe the completed observations without promoting a partial minimum."""
    labels={'sit_xl':'SiT-XL/2','raev2':'RAEv2','jit':'JiT-B/16','sit_small':'SiT-S/2（ImageNet-100）'}
    complete=report['status']['phase']=='complete'
    failed=report['status']['phase']=='failed'
    paused=report['status']['phase']=='paused_after_review'
    phase_text=('按研究价值复核提前停止，保留部分扫描' if paused else
                '扫描已结束' if complete else
                '队列因执行错误停止，扫描未完成' if failed else '扫描尚未完成')
    failures=report['numerical_failures']
    lines=['# Lifting 与普通 IG：四模型宽强度扫描', '',
        f"更新于 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。"
        f"{phase_text}："
        f"新增完整 1K 配置 {report['completed_new']}/{report['planned_new']}，"
        f"复用 {report['reused']} 份历史结果，记录 {len(failures)} 个数值失败配置。",
        '',
        '这是一组配对探索曲线。单个强度上的改善、离散网格中的最优值、充分调参后的最优质量，'
        '是不同结论；当前实验不提供独立确认或等计算成本的优越性结论。', '',
        '[冻结协议](LIFTING_WIDE_SCALE_PROTOCOL_20260909_ZH.md) · '
        '[全部数据](data/lifting_wide_scale_20260909/curves.csv) · '
        '[数值复核](data/lifting_wide_scale_20260909/audit.json) · '
        '[算子推导与强度重标定](LIFTING_MODIFIED_FLOW_AND_SCALE_20260909_ZH.md)', '',
        '![四模型强度曲线](data/lifting_wide_scale_20260909/curves.png)', '',
        '共同主网格为 α=0/.25/.5/1/1.5/2；历史锚点同时保留。普通 IG 的传统 scale 为 1+α。'
        'lifting 本身承担引导，每个活动区间写入一次；两个操作没有叠加。'
        '每个模型保持自身固定时间窗口和强度形状，因此横轴是该模型的峰值额外强度。', '',
        '## 已有结果是否足够强', '',
        '| 模型 | IG 已观察最低 FID | lifting 已观察最低 FID | lifting 相对改善 |',
        '|---|---:|---:|---:|']
    if paused:
        lines[10:10]=[
            '本轮已完成 SiT-XL 与 RAEv2 的全部预定点。经用户要求重新评估研究价值后，'
            '停止其余采样：JiT 首个新配置的部分 batch 原样保留，不计算 FID；'
            '小 SiT 的新配置未启动。下方 JiT、小 SiT 曲线来自此前的历史结果。'
            '提前停止是研究优先级决定，不是数值失败，也不表示四模型扫描完成。', '',
            '[本轮决定与 IG 承载等式研究](IG_FIXED_POINT_CARRIER_RETHINK_20260909_ZH.md)', '']
    for model in MODELS:
        observed=report['comparisons'][model]['all_observed_best']
        fmt=lambda r: f"{r['fid']:.5f}（α={r['alpha']:g}）" if r else ('未完成，已暂停' if paused else '待完成')
        ig=observed['ig'];lift=observed['lifting']
        gain=f"{100*(1-lift['fid']/ig['fid']):+.3f}%" if ig and lift else '—'
        lines.append(f"| {labels[model]} | {fmt(ig)} | {fmt(lift)} | {gain} |")
    lines.extend(['',
        '这一表回答目前是否观察到更好的最低 FID；它包含不同密度的历史扫描，'
        '不能视为对称搜索后的最终比较。小 SiT 的旧未分段 IG 另列在文末，未混入本轮重启曲线。', '',
        '较宽的平台也需要保持足够好的质量。本轮扫描前，RAEv2 已有 α=.65/.78/.85/.90 的 lifting FID '
        '为38.4523/38.4685/38.4566/38.4972，较平但均差于原生 IG38.2642；'
        'JiT 当时已有 α=.24/.30/.36 的 lifting 为55.4558/55.2236/56.0385，尚不能据三个锚点确定平台宽度。'
        '这些具体观察不替代下方宽网格尚未完成的点。', '',
        '## 共同主网格上的当前观察', '',
        '| 模型 | 共同主网格：IG 最低 FID | 共同主网格：lifting 最低 FID | 完成状态 |',
        '|---|---:|---:|---|'])
    for model in MODELS:
        comparison=report['comparisons'][model]
        fmt=lambda r: f"{r['fid']:.5f}（α={r['alpha']:g}）" if r else ('未完成，已暂停' if paused else '待完成')
        observed=comparison['common_grid_best']
        lines.append(f"| {labels[model]} | {fmt(observed['ig'])} | {fmt(observed['lifting'])} | "
                     f"{'队列完成' if comparison['complete'] else '尚未完成，最低值仅限已完成点'} |")
    lines.extend(['', '表中 α=0 是两种方法共用的 Strong 结果，未重复计入采样量。'
        '历史采样网格密度不同，因此不把“所有已观察点”的最低值混入共同主网格比较。', ''])
    for model in MODELS:
        data=[r for r in rows if r['model']==model]
        curves={method:{r['alpha']:r for r in data if r['method']==method} for method in ('ig','lifting')}
        if 0. in curves['ig']:
            curves['lifting'][0.]=curves['ig'][0.]
        strengths=sorted(set(request['alpha_grid'])|set(curves['ig'])|set(curves['lifting']))
        lines.extend([f"## {labels[model]}", '',
            '| α | 普通 IG FID ↓ | lifting FID ↓ | 同强度改善率 | lifting / IG 耗时 |',
            '|---:|---:|---:|---:|---:|'])
        def fid_text(record):
            if not record:return '未完成，已暂停' if paused else '待完成'
            if record['status']!='complete':return '数值失败'
            return f"{record['fid']:.5f}" + (' †' if record['reused'] else '')
        for alpha in strengths:
            ig=curves['ig'].get(alpha);lift=curves['lifting'].get(alpha)
            valid=ig and lift and ig['status']==lift['status']=='complete'
            gain=f"{100*(1-lift['fid']/ig['fid']):+.3f}%" if valid else '—'
            cost='—'
            if valid and alpha and ig.get('sum_batch_gpu_seconds') and lift.get('sum_batch_gpu_seconds'):
                cost=f"{lift['sum_batch_gpu_seconds']/ig['sum_batch_gpu_seconds']:.3f}×"
            lines.append(f"| {alpha:g} | {fid_text(ig)} | {fid_text(lift)} | {gain} | {cost} |")
        lines.append('')
        comparable=[p for p in report['comparisons'][model]['paired'] if p['alpha']>0]
        if comparable:
            wins=sum(p['relative_fid_gain_percent']>0 for p in comparable)
            lines.extend([f"已完成 {len(comparable)} 个非零同强度对照，其中 {wins} 个 lifting FID 较低。"
                '此计数不代表统计显著性，也不代表未测强度的行为。', ''])
    lines.extend(['## 可比范围与历史基准', '',
        '† 表示同一输入 bank 的历史结果，已核对像素、来源文件和参考统计的身份；不是新的独立抽样。'
        '耗时是各 batch 采样与解码 GPU 秒之和，不含模型加载、预检和 FID 评估；未记录的旧成本留空。', '',
        'SiT-XL 本轮主推进为 Euler100；旧 Euler115 调优结果 FID38.18548（IG scale=2.5）属于不同步数。'
        'JiT 的官方 CFG 基准 FID38.77898 仍明显强于此前内部头 IG 的55.48033，不能只赢弱头对照就称为该模型最佳采样。'
        '小 SiT 的旧未分段普通 IG 为64.85130，本轮普通 IG 特意使用与 lifting 一致的分段重启，二者不混成一条曲线。'
        '这些历史基准的原始语境见[研究状态中的既有结果](RESEARCH_STATUS.md)及'
        '[XL 多指标报告](OFFICIAL_SIT_XL_MULTIMETRIC_RESULTS_20260909_ZH.md)。', '',
        '1K 的有限样本 FID、同一 bank 上选强度、以及小 SiT 的历史批次 RNG 约定限制了外推。'
        '独立复核使用相同缓存 Inception 特征的另一套 FP64 算术，并非独立图像采样或第二特征模型。', '',
        '当前局部流推导说明，lifting 的差异包含强度依赖项和数值积分项；一维 Gaussian 中甚至可以完全重标定为普通 IG。'
        '所以即使曲线更平稳，也应先报告强度稳健性；是否存在额外质量机制需要完整曲线、计算成本与后续独立证据。', '',
        '研究目标尚未实现；本报告不把扫参、推导或局部正结果写成 ICLR 级研究成功。', ''])
    if failures:
        lines.extend(['## 数值失败', ''])
        for r in failures:
            lines.append(f"- {labels[r['model']]} / {r['arm']}：{r['failure']['error']}；"
                         f"保留 {r.get('completed_images',0)} 张已完成图像，FID 不计算。")
        lines.append('')
    if failed:
        lines.extend(['## 执行错误', '', repr(report['status'].get('error')), '',
                      '以上为已完成点的部分结果。执行错误不计为方法的数值失败，也不补造未完成配置的 FID。', ''])
    path=WORK/'docs/LIFTING_WIDE_SCALE_RESULTS_20260909_ZH.md'
    tmp=path.with_suffix('.tmp');tmp.write_text('\n'.join(lines));tmp.replace(path)


def arithmetic_audit(record):
    key=f"{record['model']}_{record['arm']}"
    cache=ROOT/'analysis'/f'{key}_arithmetic.json'
    directory=Path(record['sample_path']).parent
    if record['model']=='sit_small':
        feature_path=directory/'activations.npz'
        reference=SMALL_DATA/'adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz'
    else:
        matches=list((directory/'features').glob('*-inception.features.pt'))
        if not matches and record['model']=='raev2':
            matches=list((directory.parent/'features').glob('native_ig-*-inception.features.pt'))
        assert len(matches)==1,(directory,matches)
        feature_path=matches[0];reference=REF
    hashes=dict(features=sha(feature_path),reference=sha(reference),pixels=sha(record['sample_path']))
    assert hashes['pixels']==record['sample_sha256']
    if cache.exists():
        previous=read(cache)
        assert previous['hashes']==hashes and previous['reported_fid']==record['fid']
        return previous
    if record['model']=='sit_small':
        with np.load(feature_path) as f:x=f['pool_3'].astype(np.float64)
    else:
        value=torch.load(feature_path,map_location='cpu',weights_only=False)
        assert isinstance(value,torch.Tensor)
        x=value.numpy().astype(np.float64)
    assert x.shape==(1000,2048) and np.isfinite(x).all()
    with np.load(reference) as f:mu,cov=f['mu'].astype(np.float64),f['sigma'].astype(np.float64)
    mean,variance=fid_components(x,mu,cov)
    error=abs(mean+variance-record['fid'])
    assert error<.001,(key,error)
    result=dict(passed=True,hashes=hashes,features_path=str(feature_path),reference_path=str(reference),
                reported_fid=record['fid'],mean_term=mean,covariance_term=variance,
                independently_recomputed_fid=mean+variance,absolute_error=error,
                scope='Same cached features; independent FP64 arithmetic, not new sampling or feature extraction.')
    atomic(cache,result)
    return result


def main(require_complete=False):
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    request=read(ROOT/'request.json');status=read(ROOT/'status.json')
    if require_complete:assert status['phase']=='complete',status
    all_records={}
    for name,spec in request['models'].items():
        for arm,r in spec['reused'].items():all_records[(name,arm)]=r
    if (ROOT/'results.json').exists():
        for r in read(ROOT/'results.json'):all_records[(r['model'],r['arm'])]=r
    records=sorted(all_records.values(),key=lambda r:(MODELS.index(r['model']),r['method'],r['alpha']))
    rows=[];audits={};comparisons={}
    for record in records:
        if record['complete']:
            for path,h in record.get('source_files',{}).items():assert sha(path)==h,path
            check=arithmetic_audit(record);audits[f"{record['model']}/{record['arm']}"]=check
            row={key:record.get(key) for key in ('model','method','alpha','arm','reused','samples','fid',
                 'sum_batch_gpu_seconds','full_calls_per_image','prefix_calls_per_image','sample_path','sample_sha256')}
            row.update(status='complete',is_score=record['metrics'].get('inception_score'),
                       mean_term=check['mean_term'],covariance_term=check['covariance_term'],
                       arithmetic_error=check['absolute_error'])
        else:
            row={key:record.get(key) for key in ('model','method','alpha','arm','reused','fid')}
            row.update(status=record['status'],samples=record.get('completed_images',0))
        rows.append(row)
    columns=['model','method','alpha','arm','status','reused','samples','fid','is_score','mean_term',
             'covariance_term','arithmetic_error','sum_batch_gpu_seconds','full_calls_per_image',
             'prefix_calls_per_image','sample_path','sample_sha256']
    with (OUT/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,columns);w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.titlesize':12})
    fig,axes=plt.subplots(2,4,figsize=(16,7),gridspec_kw={'height_ratios':[1.3,1]})
    titles={'sit_xl':'SiT-XL/2','raev2':'RAEv2','jit':'JiT-B/16','sit_small':'SiT-S/2 (IN-100)'}
    colors={'ig':'#2a628f','lifting':'#c35a26'}
    for column,name in enumerate(MODELS):
        data=[r for r in rows if r['model']==name and r['status']=='complete']
        curves={method:{r['alpha']:r for r in data if r['method']==method} for method in ('ig','lifting')}
        if 0. in curves['ig']:
            curves['lifting'][0.]=dict(curves['ig'][0.],method='lifting',shared_zero=True)
        for method,label in [('ig','Ordinary IG'),('lifting','Lifting')]:
            values=sorted(curves[method].values(),key=lambda r:r['alpha'])
            if values:
                axes[0,column].plot([r['alpha'] for r in values],[r['fid'] for r in values],
                                    color=colors[method],lw=1.7,label=label)
                for reused,face in [(True,'white'),(False,colors[method])]:
                    subset=[r for r in values if bool(r['reused'])==reused]
                    axes[0,column].scatter([r['alpha'] for r in subset],[r['fid'] for r in subset],
                                          s=29,facecolors=face,edgecolors=colors[method],zorder=3)
        shared=sorted(set(curves['ig'])&set(curves['lifting']))
        paired=[dict(alpha=a,ig_fid=curves['ig'][a]['fid'],lifting_fid=curves['lifting'][a]['fid'],
                     relative_fid_gain_percent=100*(1-curves['lifting'][a]['fid']/curves['ig'][a]['fid'])) for a in shared]
        axes[1,column].axhline(0,color='#555555',lw=.8)
        if paired:
            axes[1,column].plot([r['alpha'] for r in paired],[r['relative_fid_gain_percent'] for r in paired],
                               'o-',color='#34775d',ms=4)
        common={method:[r for a,r in curves[method].items() if a in request['alpha_grid']] for method in curves}
        min_record=lambda values:min(values,key=lambda r:r['fid']) if values else None
        comparisons[name]=dict(paired=paired,
            common_grid_best={method:min_record(values) for method,values in common.items()},
            all_observed_best={method:min_record(list(values.values())) for method,values in curves.items()},
            common_grid_points={method:sorted(a for a in values if a in request['alpha_grid']) for method,values in curves.items()},
            complete=(ROOT/name/'complete.json').exists())
        axes[0,column].set_title(titles[name]);axes[0,column].set_ylabel('FID-1K (lower is better)')
        axes[1,column].set_ylabel('Lifting gain over matched IG (%)')
        for ax in axes[:,column]:
            ax.set_xlim(-.04,2.04);ax.set_xticks([0,.25,.5,1,1.5,2]);ax.grid(alpha=.18)
            ax.set_xlabel('Peak extra strength (IG scale = 1 + strength)')
        if not data:axes[0,column].text(.5,.5,'Pending',ha='center',transform=axes[0,column].transAxes)
    axes[0,0].legend(frameon=False,loc='best')
    completed_new=sum(not r['reused'] and r['complete'] for r in records)
    planned_new=sum(len(s['new_points']) for s in request['models'].values())
    finished=status['phase']=='complete'
    phase_label='Paused after review' if status['phase']=='paused_after_review' else 'Completed' if finished else 'Partial'
    fig.suptitle(f"{phase_label}: {completed_new}/{planned_new} new 1K configurations",fontsize=14)
    fig.text(.5,.018,'Open markers: reused results. Each model uses one fixed input bank and time profile. No independent confirmation or compute matching.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(OUT/'curves.png',dpi=180);fig.savefig(OUT/'curves.pdf');plt.close(fig)
    report=dict(status=status,completed_new=completed_new,planned_new=planned_new,reused=len(records)-sum(not r['reused'] for r in records),
                request_sha256=sha(ROOT/'request.json'),arithmetic_audits=audits,comparisons=comparisons,
                numerical_failures=[r for r in records if not r['complete']],research_goal_achieved=False)
    atomic(OUT/'audit.json',report)
    render_report(request,report,rows)
    print(f"Audited {len(audits)} completed banks; {completed_new}/{planned_new} new configurations completed.",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    main(args.require_complete)
