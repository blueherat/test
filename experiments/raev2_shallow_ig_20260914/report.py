"""Source, pairing, cost and FID audit for all searched coefficients and held-out 5K."""
import csv,json,re,shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from experiments.raev2_shallow_ig_20260914 import sample as s
from experiments.raev2_shallow_ig_20260914 import grid_common as grid_storage
from experiments.guidance_pasted_20260912.audit import fid_from_features
from experiments.raev2_context_5k_20260914.report import reference_moments,fid_from_covariance
c,m=s.c,s.m
OUT=c.WORK/'docs/data/raev2_shallow_ig_50k_20260914'
REPORT=c.WORK/'docs/RAEV2_SHALLOW_IG_50K_RESULTS_20260914_ZH.md'
NAMES={'incumbent':'Native depth8','native4':'DDT head depth4 50K','mlp4':'MLP depth4 50K','mlp8':'MLP depth8 50K'}


def audit(stage,reference,*,configs=None,require_controller_complete=True):
    request=s.verify(stage);root=m.ROOT/stage
    if require_controller_complete:assert c.read(root/'controller_complete.json')['complete']
    selected=request['configs'] if configs is None else configs
    assert all(cfg in request['configs'] for cfg in selected)
    noise=np.load(request['noise'],mmap_mode='r');labels=np.load(request['labels']);rh=c.sha(root/'request.json')
    if request['samples']==5000:assert np.array_equal(np.bincount(labels,minlength=1000),np.full(1000,5))
    else:assert len(np.unique(labels))==request['samples']
    rows=[];audits=[];out=OUT/stage;out.mkdir(parents=True,exist_ok=True)
    for cfg in selected:
        folder=root/cfg['arm']
        if (folder/'invalid.json').exists():
            failure=c.read(folder/'invalid.json');assert failure['invalid'] and failure['request_sha256']==rh
            shutil.copy2(folder/'invalid.json',out/f"{cfg['arm']}_invalid.json");continue
        summary=c.read(folder/'summary.json');metric=c.read(folder/'metrics.json')[0]
        assert summary['complete'] and summary['samples']==request['samples'] and summary['request_sha256']==rh
        assert c.sha(folder/'samples.npz')==summary['samples_sha256']==metric['sample_sha256']
        with np.load(folder/'samples.npz') as d:images=d['arr_0']
        assert images.shape==(request['samples'],256,256,3) and images.dtype==np.uint8
        coverage=[];seconds=0.;expected=99 if cfg['kind']!='incumbent' and cfg['w']!=1 else 0
        for record in summary['records']:
            path=Path(record['path']);assert c.sha(path)==record['sha256']
            with np.load(path) as d:
                start=int(d['start']);stop=start+len(d['pixels']);coverage.extend(range(start,stop))
                assert str(d['noise_sha256'])==c.array_sha(noise[start:stop]) and str(d['request_sha256'])==record.get('source_request_sha256',rh)
                assert np.array_equal(d['labels'],labels[start:stop]) and np.array_equal(d['pixels'],images[start:stop])
                grid_storage.check_latents(d,(stop-start,1024,16,16))
                assert np.array_equal(d['block_calls'],np.full(30,100)) and int(d['full'])==100 and int(d['head_calls'])==expected
                seconds+=float(d['seconds'])
        assert coverage==list(range(request['samples'])) and abs(seconds-summary['seconds'])<1e-6
        paths=list((folder/'features').glob('*.features.pt'));assert len(paths)==1
        features=torch.load(paths[0],map_location='cpu',weights_only=True).numpy()
        assert features.shape==(request['samples'],2048) and np.isfinite(features).all()
        recalculated=(fid_from_features(features,reference[0],reference[1]) if request['samples']<2048 else fid_from_covariance(features,reference))
        error=abs(recalculated-metric['fid']);assert error<.002,(cfg,error)
        audits.append(dict(arm=cfg['arm'],passed=True,coverage=request['samples'],fid_fp64=recalculated,error=error,
            independent_extractor=False,samples_sha256=summary['samples_sha256'],features_sha256=c.sha(paths[0])))
        row=dict(cfg,stage=stage,samples=request['samples'],fid=metric['fid'],inception_score=metric['inception_score'],
            full_calls=100,prefix_calls=0,head_calls=expected,seconds=seconds,source_output=str(folder))
        rows.append(row)
        if stage=='confirm_5k':
            image=Image.new('RGB',(6*160,180),'white');draw=ImageDraw.Draw(image)
            for j,pixel in enumerate(images[:6]):
                image.paste(Image.fromarray(pixel).resize((156,156)),(j*160,0));draw.text((j*160+2,157),f'ID {j}; class {labels[j]}',fill='black')
            image.save(out/f"{cfg['arm']}.png")
        shutil.copy2(folder/'metrics.json',out/f"{cfg['arm']}_metrics.json")
    c.atomic(out/'results.json',dict(complete=True,rows=rows,audits=audits,request_sha256=rh,
        audit_scope='whole_stage' if configs is None else 'explicit_completed_subset',
        requested_config_count=len(request['configs']),audited_config_count=len(selected)))
    shutil.copy2(root/'request.json',out/'request.json');print('Audited',stage,len(rows),flush=True)
    return rows,audits


def plot_search(all_rows,chosen):
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    tuning=[r for r in all_rows if r['stage'] in ('balanced_1000','balanced_refine_1000')]
    for kind in s.FAMILIES:
        rows=sorted([r for r in tuning if r['kind']==kind or r['w']==1],key=lambda x:x['w'])
        for ax in axes:
            line=ax.plot([r['w'] for r in rows],[r['fid'] for r in rows],'-o',label=NAMES[kind],markersize=3)[0]
            ax.scatter([chosen[kind]['best_w']],[chosen[kind]['tuning_fid']],marker='*',s=100,color=line.get_color(),edgecolor='black',linewidth=.4,zorder=5)
    baseline=next(r['fid'] for r in tuning if r['w']==1)
    for ax in axes:
        ax.set(xlabel='Extrapolation coefficient w',ylabel='FID (1000 balanced paired images)');ax.grid(alpha=.2)
    axes[0].set_title('Complete finite scan');axes[0].legend(fontsize=7)
    axes[1].set(title='Zoom near selected coefficients',xlim=(.98,min(3,max(v['best_w'] for v in chosen.values())+.25)),
        ylim=(min(r['fid'] for r in tuning)-.15,baseline+1.0))
    fig.tight_layout();fig.savefig(OUT/'coefficient_search.png',dpi=160);fig.savefig(OUT/'coefficient_search.pdf');plt.close(fig)


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);reference=reference_moments()
    assert c.read(m.ROOT/'pipeline_complete.json')['complete']
    all_rows=[];all_audits=[]
    for stage in ('tune_native','balanced_1000','balanced_refine_1000','confirm_5k'):
        rows,audits=audit(stage,reference);all_rows+=rows;all_audits+=audits
    fields=['stage','arm','kind','w','samples','fid','inception_score','full_calls','prefix_calls','head_calls','seconds','source_output']
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(all_rows)
    chosen=c.read(m.ROOT/'selected_coefficients.json');shutil.copy2(m.ROOT/'selected_coefficients.json',OUT/'selected_coefficients.json')
    plot_search(all_rows,chosen)
    final=[r for r in all_rows if r['stage']=='confirm_5k'];native=next(r for r in final if r['kind']=='incumbent')
    canvas=Image.new('RGB',(250+6*160,35+len(final)*190),'white');draw=ImageDraw.Draw(canvas)
    draw.text((8,8),'First six fixed paired IDs; no sample selection',fill='black')
    for i,row in enumerate(final):
        top=35+190*i;draw.text((8,top+10),NAMES[row['kind']],fill='black')
        draw.text((8,top+30),f"w={row['w']:.3f}; FID={row['fid']:.3f}",fill='black')
        canvas.paste(Image.open(OUT/'confirm_5k'/f"{row['arm']}.png"),(250,top))
    canvas.save(OUT/'paired_5k.png')
    training={}
    for name,root in [('depth4',m.TRAIN),('depth8',s.d8.TRAIN)]:
        summary=c.read(root/'summary.json');assert summary['steps']==50000 and summary['strong_frozen']
        assert c.sha(root/'head.pt')==summary['head_sha256']
        checkpoint=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
        retained=torch.load(root/'head.pt',map_location='cpu',weights_only=True)
        assert checkpoint['step']==retained['step']==50000 and checkpoint['request_sha256']==c.sha(root/'request.json')
        assert [item['step'] for item in checkpoint['history']]==list(range(1,50001))
        active=('native','mlp') if name=='depth4' else ('context',)
        for key in active:
            assert all(float(v['step'])==50000 for v in checkpoint['optimizers'][key]['state'].values())
            assert all(torch.equal(v,retained['ema'][key][k]) for k,v in checkpoint['ema'][key].items())
        c.atomic(OUT/f'{name}_training_audit.json',dict(passed=True,step=50000,history_contiguous=True,
            optimizer_counters=50000,ema_exact=True,checkpoint_sha256=c.sha(root/'latest.pt')))
        del checkpoint,retained
        training[name]=summary
        shutil.copy2(root/'request.json',OUT/f'{name}_training_request.json')
        shutil.copy2(root/'summary.json',OUT/f'{name}_training_summary.json')
    assert training['depth4']['strong_state_sha256_before']==training['depth4']['strong_state_sha256_after']
    assert c.read(s.d8.TRAIN/'resume_verified.json')['passed']
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for key,label in [('native','DDT depth4'),('mlp','MLP depth4')]:
        values=[c.read(p) for p in sorted((m.TRAIN/'validation').glob('step*.json'))]
        axes[0].plot([v['step'] for v in values],[v['means'][key] for v in values],'-o',label=label,markersize=3)
    values={}
    for folder in (s.d8.PREVIOUS/'fixed_validation',s.d8.TRAIN/'fixed_validation'):
        for p in sorted(folder.glob('step_*.json')):
            v=c.read(p);values[v['step']]=v['splits']['validation']['mean']
    points=sorted(values.items());axes[1].plot([x[0] for x in points],[x[1] for x in points],'-o',label='MLP depth8',markersize=3)
    axes[1].axvline(20000,ls='--',color='gray',label='Resume at20K')
    for ax in axes:ax.set(xlabel='Training steps',ylabel='Fixed validation clean-latent MSE');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Separate fixed validation noise banks; use within-head trajectories')
    fig.tight_layout();fig.savefig(OUT/'training_validation.png',dpi=150);plt.close(fig)
    best=min(final,key=lambda r:r['fid']);change=best['fid']-native['fid']
    best_head=min([r for r in final if r['kind']!='incumbent'],key=lambda r:r['fid'])
    original=c.read(s.LEGACY/'native_base/metrics.json')
    if isinstance(original,list):original=original[0]
    original_delta=best_head['fid']-original['fid']
    lines=['# RAEv2：50K冻结后训练头、外推系数搜索及独立5K','',
        f"已完成。四种设置分别在均衡1K上选取外推系数，再生成独立5K。后训练头中最低FID来自 **{NAMES[best_head['kind']]}，w={best_head['w']:.3f}，FID={best_head['fid']:.4f}**。相对默认原生IG（w=1.78，既有同bank FID={original['fid']:.4f}）差值为 **{original_delta:+.4f}**；相对本次重新选参的原生IG差值为{best_head['fid']-native['fid']:+.4f}。负数才是改善。",'',
        '原生IG位于28层编码器的第8层，强分支另有2层DDT decoder。本轮新增第4层同结构DDTFinalLayer和Context MLP，各50K；既有第8层MLP从完整20K在线／EMA／AdamW／RNG状态续训至50K。所有主干权重冻结。第4层两种头参数数分别为5,627,104与5,635,744，训练输入、噪声与时间完全相同。第8层保留原训练随机流。','',
        '外推统一为weak+w×(strong−weak)，w=1是纯strong，当前原生默认w=1.78。扫描的是采样外推系数；每个头在共享类别均衡1000图bank上作粗网格和邻域中点细化，5K使用独立噪声bank且不再调参。因此“最优”仅指本次有限搜索与筛选样本上的最低FID，不能声称连续全局最优。','',
        '![完整系数扫描](data/raev2_shallow_ig_50k_20260914/coefficient_search.png)','',
        '| 设置 | 选定w | 筛选FID（1000） | 独立5K FID↓ | IS↑ | 相对已调参原生IG | 相对默认原生IG |','|---|---:|---:|---:|---:|---:|---:|']
    for row in final:
        lines.append(f"| {NAMES[row['kind']]} | {row['w']:.3f} | {chosen[row['kind']]['tuning_fid']:.3f} | {row['fid']:.4f} | {row['inception_score']:.3f} | {row['fid']-native['fid']:+.4f} | {row['fid']-original['fid']:+.4f} |")
    invalid=[]
    for stage in ('balanced_1000','balanced_refine_1000'):
        for p in sorted((m.ROOT/stage).glob('*/invalid.json')):invalid.append(dict(c.read(p),stage=stage))
    lines+=['','最初400图原生扫描只覆盖400类；其结果保留作诊断，所有候选统一补成1000类均衡扫描后选参。已完成的原始400图批次逐位复用，缺失批次和剩余600图生成后合并，来源分别记录。','']
    if invalid:lines+=['数值稳定性检查失败、未报部分样本FID的候选：'+ '；'.join(f"{x['config']['kind']} w={x['config']['w']} ({x['stage']}, {x['error']})" for x in invalid)+'。原采样器同时检查非有限值和状态绝对值大于1e6；异常消息未区分两种触发条件，因此不把它一律解释为NaN。','']
    legacy=[]
    for arm,desc in [('native_base','原生IG固定w=1.78'),('context20k_half','20K第8层MLP固定w=1.39')]:
        metric=c.read(s.LEGACY/arm/'metrics.json')
        if isinstance(metric,list):metric=metric[0]
        legacy.append(dict(arm=arm,description=desc,fid=metric['fid'],source=str(s.LEGACY/arm),reused=True))
    lines+=['','同一个5K bank还有明确复用的历史参考：'+ '；'.join(f"{r['description']}：FID {r['fid']:.4f}" for r in legacy)+'。这些历史结果不是本次新生成或独立重复；不能由不同w的20K/50K差值单独归因于训练步数。','',
        '采样均为Euler100、shift8、活动窗口[.1,1]、FP32 velocity、BF16模型、batch16；每图100次全主干调用，无额外prefix。w>1时新头每图调用99次，原生弱头仍随主干计算。部分已完成的1K和5K指标计算曾与其余头的采样在GPU0重叠，记录的墙钟时间不作为独占GPU速度基准；计算量比较使用实际记录的全主干／前缀／读出头调用数。旧原生首16张latent和像素逐位重放通过；新头零外推与共享特征、全部30个block调用数也核对。','',
        '训练使用互斥的5,000个真实训练latent与1,000个验证latent，50K终点提前固定。第4层验证噪声与第8层沿用的验证噪声不同，误差记录用于各自收敛诊断，不据此作跨层因果比较。头容量、深度及初始化／训练历史存在差别，生成测试评价完整配置，不证明单一结构因素造成收益。','',
        '5K共覆盖1000类、每类5图；所有配置配对，未筛图。采用相同Nanogen FP32 Inception、batch64和参考统计，保存特征并重算FP64 FID。重算验证算术，仍使用同一个特征提取器。本轮只有一个5K生成bank；细小差异未获得独立重复或显著性保证。','',
        '![验证误差轨迹](data/raev2_shallow_ig_50k_20260914/training_validation.png)','',
        '![固定前6个ID](data/raev2_shallow_ig_50k_20260914/paired_5k.png)','',
        '[完整结果CSV](data/raev2_shallow_ig_50k_20260914/results.csv) · [选参记录](data/raev2_shallow_ig_50k_20260914/selected_coefficients.json) · [训练协议](RAEV2_SHALLOW_IG_50K_PROTOCOL_20260914_ZH.md) · [均衡1K补充协议](RAEV2_SHALLOW_IG_BALANCED_SEARCH_20260914_ZH.md) · [实现](../experiments/raev2_shallow_ig_20260914/core.py)。','',
        f'原始完整结果：`{m.ROOT}`。']
    if (m.ROOT/'planned_5k_candidates.json').exists():
        lines[1:1]=['', '> 本页保留原 1K 选参阶段结果。原 confirm_5k 现已重新归为系数筛选数据；修订方案以另一个新种子做最终确认。[最新 5K 选参报告](RAEV2_SHALLOW_IG_5K_SELECTION_RESULTS_20260914_ZH.md)。']
    REPORT.write_text('\n'.join(lines)+'\n')
    unique_records={}
    for row in all_rows:
        for record in c.read(Path(row['source_output'])/'summary.json')['records']:unique_records[record['path']]=record
    unique_images=0
    for path in unique_records:
        with np.load(path) as d:unique_images+=len(d['pixels'])
    result=dict(complete=True,rows=all_rows,audits=all_audits,legacy=legacy,selected=chosen,invalid=invalid,best_head_delta_original_native=original_delta,
        distinct_images_in_audited_batches=unique_images,final_new_5k_images=sum(r['samples'] for r in final),
        max_fid_error=max(a['error'] for a in all_audits),report_sha256=c.sha(REPORT))
    c.atomic(OUT/'results.json',result);c.atomic(m.ROOT/'report_complete.json',dict(complete=True,report=str(REPORT),report_sha256=c.sha(REPORT)))
    print('Report complete',[(r['kind'],r['w'],r['fid']) for r in final],flush=True)

if __name__=='__main__':main()
