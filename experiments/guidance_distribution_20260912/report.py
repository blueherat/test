"""Source-backed artifacts for the shared-contamination experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from experiments.guidance_pasted_20260912 import common as c
from . import mixture as m

OUT=c.WORK/'docs/data/guidance_distribution_20260912'
REPORT=c.WORK/'docs/SHARED_CONTAMINATION_EXPERIMENTS_20260912_ZH.md'
NAMES={'sit_small':'SiT-S/2','raev2':'RAEv2'}
ARMS={'inverse':'Inverse (candidate)','time_mean':'Source time mean','moment_constant':'Moment constant',
      'native_base':'Native','native_half':'Native half','constant_cap2':'Constant 2'}


def frames():
    quality=[];moment=[];events=[];filters=[];sources=[];training=[];timing=[]
    for model in c.MODELS:
        mm=c.read(m.ROOT/model/'moment_contamination/results.json')
        for row in mm['rows']:
            moment.append({k:row[k] for k in ('model','track','kappa_unconstrained',
                'heldout_mmd2_strong','heldout_mmd2_subtracted','heldout_fraction_equation_residual')}|
                dict(kappa_lo=row['kappa_resampling_2p5_97p5'][0],kappa_hi=row['kappa_resampling_2p5_97p5'][1],
                     change_lo=row['corrected_change_resampling_2p5_97p5'][0],
                     change_hi=row['corrected_change_resampling_2p5_97p5'][1]))
        for track in ('ig','cfg'):
            p=m.ROOT/model/m.stage(track)/'results.json'
            assert p.exists(),p
            rows=c.read(p);assert len(rows)==6,(p,len(rows))
            best=min(r['fid'] for r in rows if r['arm']!='inverse')
            native=next(r for r in rows if r['arm']=='native_base')
            for row in rows:
                record={k:row[k] for k in ('model','track','arm','fid','inception_score','mean_gain',
                    'invalid_rho_fraction','cap_triggered_fraction','seconds','full_calls_per_output',
                    'prefix_calls_at_inference','primary_samples','generated_paths')}
                record['delta_vs_best_control']=row['fid']-best
                record['passes_fixed_screen']=row['arm']=='inverse' and row['fid']<=best-2 and row['inception_score']>=.9*native['inception_score']
                quality.append(record)
            ss=c.read(m.ROOT/model/(track+'_contamination_profile')/'summary.json')
            events.append(dict(model=model,track=track,kappa=ss['kappa'],
                signed_event_mean=ss['signed_event_mean'],
                lower=ss['signed_event_resampling_2p5_97p5'][0],upper=ss['signed_event_resampling_2p5_97p5'][1],
                strong_event_frequency=ss['stats']['strong']['invalid_probability_fraction'],
                reference_event_frequency=ss['stats']['reference']['invalid_probability_fraction'],
                strong_profile_mean_gain=ss['stats']['strong']['mean_gain']))
            ff=c.read(m.ROOT/model/(track+'_endpoint_filter')/'summary.json')
            filters.append({k:ff[k] for k in ('model','track','kappa','mmd2_before','mmd2_after_weighted','mmd2_change',
                'mean_accept_probability','ideal_accept_probability','effective_sample_size',
                'predicted_negative_accept_fraction','source_audit_bce','source_audit_accuracy')}|
                dict(change_lo=ff['change_resampling_2p5_97p5'][0],change_hi=ff['change_resampling_2p5_97p5'][1]))
        for r in c.read(m.ROOT/model/'endpoint_sources/results.json'):
            sources.append({k:r[k] for k in ('model','arm','primary_samples','fid','inception_score','seconds',
                'full_calls_per_output','prefix_calls_per_output')})
        d=c.read(m.ROOT/model/'cfg_source_data/training_complete.json')
        training.append(dict(model=model,track='cfg',inverse_temperature=d['inverse_temperature'],
            audit_nll=d['audit_calibrated']['bce'],audit_accuracy=d['audit_calibrated']['accuracy'],
            source_seconds=d['source_seconds'],head_fit_cpu_seconds=d['train_seconds'],
            trainable_parameters=d['trainable_parameters']))
        p=m.ROOT/model/'inference_benchmark.json'
        if p.exists():
            for d in c.read(p)['records']:
                timing.append(dict(model=model,track=d['track'],batch=d['batch'],
                    native_seconds=d['median_seconds']['original_native'],candidate_seconds=d['median_seconds']['inverse'],
                    relative_change=d['relative_median_change'],repeats=d['repetitions'],
                    native_parity=d['original_native_output_exact'],parameters=d['parameter_count']))
    return dict(quality=pd.DataFrame(quality),moments=pd.DataFrame(moment),source_events=pd.DataFrame(events),
                endpoint_filter=pd.DataFrame(filters),endpoint_sources=pd.DataFrame(sources),
                cfg_classifier=pd.DataFrame(training),inference_timing=pd.DataFrame(timing))


def plots(tables):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,model in enumerate(c.MODELS):
        for j,track in enumerate(('cfg','ig')):
            ax=axes[i,j];df=tables['quality'].query('model == @model and track == @track').set_index('arm').loc[list(m.KINDS)]
            yy=np.arange(len(df));colors=['#176b8d']+['#a3a9ae']*5
            ax.barh(yy,df.delta_vs_best_control,color=colors)
            ax.set_yticks(yy,[ARMS[k] for k in df.index]);ax.invert_yaxis()
            ax.axvline(0,color='#333333',lw=.8);ax.axvline(-2,color='#176b8d',lw=1,ls='--')
            for y,(arm,row) in zip(yy,df.iterrows()):
                ax.text(max(row.delta_vs_best_control,0)+.25,y,f'FID {row.fid:.2f}',va='center',fontsize=9)
            ax.set_xlim(min(-3,float(df.delta_vs_best_control.min())-1),max(4,float(df.delta_vs_best_control.max())+8))
            ax.set_title(NAMES[model]+' · '+track.upper()+' · 400 images')
            ax.set_xlabel('FID difference from best fixed control (lower is better)')
    fig.suptitle('Shared-contamination inverse: candidate and all amplitude/time controls',fontsize=14)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'quality_comparison.{ext}',dpi=180)
    plt.close(fig)
    mm=tables['moments'];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    yy=np.arange(len(mm));labels=[NAMES[r.model]+' '+r.track for r in mm.itertuples()]
    axes[0].errorbar(mm.kappa_unconstrained,yy,xerr=np.vstack((mm.kappa_unconstrained-mm.kappa_lo,
        mm.kappa_hi-mm.kappa_unconstrained)),fmt='o',color='#176b8d',capsize=4)
    axes[0].axvline(0,color='gray',ls='--');axes[0].set_yticks(yy,labels);axes[0].invert_yaxis()
    axes[0].set_title('One fitted proportion per model and track')
    axes[0].set_xlabel('Kappa; paired-seed resampling range (descriptive)')
    ff=tables['endpoint_filter']
    signed=[];filtered=[]
    for r in mm.itertuples():
        x=ff[(ff.model==r.model)&(ff.track==r.track.lower())]
        signed.append(r.heldout_mmd2_subtracted/r.heldout_mmd2_strong)
        filtered.append(float(x.iloc[0].mmd2_after_weighted/x.iloc[0].mmd2_before))
    axes[1].barh(yy-.15,signed,height=.28,label='Signed moment subtraction',color='#176b8d')
    axes[1].barh(yy+.15,filtered,height=.28,label='Positive endpoint weights',color='#a3a9ae')
    axes[1].axvline(1,color='gray',ls='--');axes[1].set_yticks(yy,labels);axes[1].invert_yaxis()
    axes[1].set_xlabel('Held-out kernel discrepancy / unfiltered Strong estimate')
    axes[1].set_title('Feature-distribution diagnostics; these are not FID')
    axes[1].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.17))
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'contamination_evidence.{ext}',dpi=180)
    plt.close(fig)


def grids(tables):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for model in c.MODELS:
        for track in ('ig','cfg'):
            size=192;left=205;top=62;rh=214
            canvas=Image.new('RGB',(left+size*4,top+rh*6),'white');d=ImageDraw.Draw(canvas)
            d.text((12,8),NAMES[model]+' '+track.upper()+': first four fixed sample IDs; no selection',fill='black',font=font)
            labels=np.load(m.ROOT/model/m.stage(track)/'inputs/labels.npy')[:4]
            for j,label in enumerate(labels):d.text((left+j*size+8,37),f'ID {j}, class {label}',fill='black',font=small)
            for i,kind in enumerate(m.KINDS):
                root=m.ROOT/model/m.stage(track)/kind
                metric=c.read(root/'metrics.json')
                with np.load(root/'samples.npz') as f:images=f['arr_0'][:4]
                d.text((10,top+i*rh+45),ARMS[kind],fill='black',font=small)
                d.text((10,top+i*rh+70),f'400-sample FID {metric["fid"]:.2f}',fill='black',font=small)
                for j,pixels in enumerate(images):
                    canvas.paste(Image.fromarray(pixels).resize((size,size),Image.Resampling.LANCZOS),(left+j*size,top+i*rh))
            canvas.save(OUT/f'{model}_{track}_first4.png')


def write_report(t):
    q=t['quality'];eligible=q[q.passes_fixed_screen]
    text=[
        '# 共同污染假设：真实分布、直接采样和当前结论\n',
        '**这条假设仍是主线。** 本轮没有把局部参考替代成主线：其两个模型的训练产物保留，生成全部暂缓。'
        '本报告把同一个共同污染假设落实为端点分布检验、固定比例的自适应采样，以及一个使用既有图像的离线过滤诊断。'
        '没有重启蒸馏、旧大队列或新的参数网格。\n',
        '## 最直接的生成结果\n',
        '两个模型、CFG/IG 两条主线、每组1个候选和5个固定强度／时间对照，共24臂，每臂400个独立图像。'
        '同组使用相同噪声与类别；没有第二配对分支。400样本只做筛选，RAE只覆盖400类，不能据此声称统计显著。'
        '候选需比全部控制至少好2 FID且IS不低于native的90%才进入全新1K确认。\n',
        f'本轮满足该规则的组数：**{len(eligible)}/4**。没有通过的组不自动进入1K/5K。'
        '不能把优于原幅度、但与另一个常数幅度几乎相同，写成自适应机制成功。\n',
        '|模型／主线|去污染候选 FID|原 guidance FID|最好固定控制|候选减最好控制|是否通过|\n'
        '|---|---:|---:|---|---:|---|\n']
    for model in c.MODELS:
        for track in ('cfg','ig'):
            df=q[(q.model==model)&(q.track==track)];candidate=df[df.arm=='inverse'].iloc[0]
            best=df[df.arm!='inverse'].sort_values('fid').iloc[0];native=df[df.arm=='native_base'].iloc[0]
            text.append(f'|{NAMES[model]} {track.upper()}|{candidate.fid:.4f}|{native.fid:.4f}|'
                        f'{ARMS[best.arm]}：{best.fid:.4f}|{candidate.fid-best.fid:+.4f}|'
                        f'{"是" if candidate.passes_fixed_screen else "否"}|\n')
    text += [
        '\n![候选与全部固定对照](data/guidance_distribution_20260912/quality_comparison.png)\n',
        '[完整24臂 CSV](data/guidance_distribution_20260912/quality.csv)。全量IS、实际使用的平均引导强度、保护触发比例也在表中，'
        '没有按FID挑选或删掉差的配置。\n',
        '## 实际检验的是哪个公式\n',
        '原假设为 $p_S=(1-a)p_\\star+a p_e$、$p_W=(1-b)p_\\star+b p_e$，$0\\le a<b\\le1$。'
        '消去共同误差分布后，$p_S=(1-\\kappa)p_\\star+\\kappa p_W$，其中 $\\kappa=a/b$。'
        '固定时间与条件时，精确 score 形式为\n\n'
        '$$s_\\star=s_S+\\frac{\\rho}{1-\\rho}(s_S-s_W),\\qquad \\rho=\\kappa\\frac{p_W}{p_S}.$$\n\n'
        '若改用真实采样边缘与概率流，且 $q_S-\\kappa q_W>0$ 在全程成立、$\\kappa$ 为常数，'
        '相同系数也由连续性方程的线性性给出。这是本轮真实来源头的依据。'
        '它不是“任意训练网络都是精确 score”的假定。完整推导、空间／时间变系数的附加项和误差共线问题见'
        '[推导报告](AG_IG_SHARED_CONTAMINATION_DERIVATION_20260912_ZH.md)。\n',
        '本轮 R 的定义对 IG 是窗口内用原弱场、窗口外用 Strong；对 CFG 是窗口内用 null、窗口外用 Strong。'
        'SiT 的 IG/CFG 窗口分别是生成前半程／前四分之三；RAE 沿用原 IG 噪声时间[.1,1]和全程CFG。'
        '不能把这些端点偷换为全程独立弱模型的分布。\n',
        '公式与已有 FBG 的去污染代数同型，FBG 也使用最大引导保护；本轮不据此声称新颖公式。'
        'FBG 的实现还引入路径后验近似、温度和偏置，并搜索引导参数。这里固定比例来自新数据，'
        '密度比来自来源分类，不复现其调参结果。[FBG §§3.2–3.4](https://arxiv.org/html/2506.06085v2)\n',
        '## 端点矩：SiT有可重复迹象，RAE仍不确定\n',
        '每个模型新收集Strong、Weak、Null各1000个图像，共6000个端点。前500对拟合一个全局比例，'
        '后500对检验；比例不看之后的生成FID。使用固定二次核的一、二阶Inception特征矩，'
        '核尺度仅由官方参考二阶矩决定。排除同一噪声对的核对角项，避免共享噪声的交叉估计偏差。'
        '参考 covariance 按固定目标总体矩处理。它是必要的矩关系，不能识别完整图像密度。\n',
        '|模型／主线|拟合κ|κ重采样范围|Strong留出MMD²|相减后MMD²|\n'
        '|---|---:|---|---:|---:|\n']
    for r in t['moments'].itertuples():
        text.append(f'|{NAMES[r.model]} {r.track}|{r.kappa_unconstrained:.5f}|'
                    f'[{r.kappa_lo:.5f}, {r.kappa_hi:.5f}]|{r.heldout_mmd2_strong:.6g}|{r.heldout_mmd2_subtracted:.6g}|\n')
    text += [
        '\nRAE 的比例重采样范围跨过0，核误差变化范围也跨过0，不能把点估计改善当成可靠正结果。'
        'SiT 的迹象较明确，但“矩上可以相减”不保证非负密度、条件正确性或采样收益。'
        '重采样以完整配对seed为单位；固定类别平衡使其只是描述性范围，不是严格校准的置信区间。\n',
        '![比例估计与两种端点诊断](data/guidance_distribution_20260912/contamination_evidence.png)\n',
        '## 正性：不要求分类器精确，也能检查一条必要关系\n',
        '对任意事件 $A_t$，严格共同混合必须满足 $Q_S(A_t)-\\kappa Q_R(A_t)\\ge0$。'
        '用冻结分类器定义 $A_t=\\{\\widehat q_S/\\widehat q_R<\\kappa\\}$ 后，直接在独立来源审计路径上数事件频率。'
        '判据用的是真实两源频率，不把分类器概率当真实密度；同一seed的所有时间先聚合后重采样。\n',
        '|模型／主线|平均有符号事件质量|重采样范围|\n|---|---:|---|\n']
    for r in t['source_events'].itertuples():
        text.append(f'|{NAMES[r.model]} {r.track.upper()}|{r.signed_event_mean:+.5f}|[{r.lower:+.5f}, {r.upper:+.5f}]|\n')
    text += [
        '\nSiT 的结果对“把端点拟合的同一κ直接放到全程实际采样边缘”给出反面证据。'
        '这不等于否定端点共同污染、允许近似误差的模型或给定条件的更细假设；'
        '也不能据此把所有问题归咎于分类器不准。真实ODE轨迹边缘与对生成端点做共同前向加噪所得的边缘，是不同的密度族。\n',
        '## 数值保护和来源头的边界\n',
        '唯一候选为 $\\tilde\\rho=\\min(\\hat\\rho,2/3)$、$\\gamma=\\tilde\\rho/(1-\\tilde\\rho)$。'
        '固定上限2防止估计比值跨越极点，没有搜索上限。它是数值保护，不能当成混合定理。'
        '第一查询强制log ratio=0以符合共同初始噪声。分类器读共享浅层token的均值；'
        '只能直接估计这份特征的来源比，特征充分性未经证明。全局温度校准也不提供逐状态的密度比保证。\n',
        'IG复用上一轮冻结、校准后的来源头。CFG新采800对来源路径，600训练、100校准、100审计。'
        '两源均在同一条件前缀特征上分类；null路径的额外条件前缀只在离线收集时使用。'
        'RAE这800个源只覆盖800类，训练/校准/审计的类别不重合，限制了来源头泛化。'
        'RAE CFG 的来源审计接近随机分类，故该分支的负结果不能单独检验精确比值下的理论。\n',
        t['cfg_classifier'].to_markdown(index=False)+'\n',
        '\n时间均值控制取来源审计的Strong路径均值，不取候选轨迹均值；因此它不保证新引导轨迹上的总剂量恰好匹配。'
        '表中的实际平均强度揭示这种差异，不能把候选的自衰减自动解释成质量校正。\n',
        '## 离线终点过滤：保留假设的线索，不能当成部署方法\n',
        '另用现有端点的前400对训练固定正则的来源逻辑回归，100对校准，后500对评估。'
        '采用接受权重 $[1-\\kappa\\widehat q_R/\\widehat q_S]_+$；不生成新图像。'
        '此处用Inception仅为离线诊断，不进入部署。表中是自归一加权的核误差，不能冒充等样本量普通FID。\n',
        '|模型／主线|未过滤MMD²|加权后MMD²|平均接受权重|理想接受率1−κ|有效样本量|\n'
        '|---|---:|---:|---:|---:|---:|\n']
    for r in t['endpoint_filter'].itertuples():
        text.append(f'|{NAMES[r.model]} {r.track.upper()}|{r.mmd2_before:.6g}|{r.mmd2_after_weighted:.6g}|'
                    f'{r.mean_accept_probability:.4f}|{r.ideal_accept_probability:.4f}|{r.effective_sample_size:.1f}|\n')
    text += [
        '\nSiT 的加权误差下降为这条分布思路提供了额外线索，但接受率与严格预测有较大差距，且需要负权截断。'
        'RAE的变化仍处于较大重采样不确定性之内。过滤即使有效，也会增加每个接受输出的期望生成成本，'
        '因此没有将其升级为方法候选。\n',
        '## 一个需要修正的识别问题：CFG不能只看图片边缘\n',
        'CFG 的共同污染应在 $(x,c)$ 联合分布或固定 $c$ 下判断。若完美条件模型满足'
        '$\\sum_c\\pi_c p_\\star(x\\mid c)=p_\\star(x)$，而null也精确给出 $p_\\star(x)$，'
        '则汇总图片后，Strong、Null、目标完全相同；任意κ都满足汇总矩等式。'
        '但对有信息的条件，原条件模型已完美时，条件污染比例只能为0。'
        '因此仅用汇总FID特征不能识别条件污染；本轮κ是探索性全局估计，RAE的不稳定结果与这个问题相容。\n',
        '这也是继续保留主假设时必须优先修正的地方：保持一个全局κ，但在包含类别匹配的联合分布上估计与检验，'
        '再决定是否需要更充分的密度比读出。不是给1000个类别各调一个系数，也不是把来源分类正确率改名为质量。'
        '本轮没有用已经看到的质量结果回头改κ，亦未暗中运行这项后续修订。\n',
        '## 推理成本与执行检查\n',
        'SiT IG/CFG 每个输出分别为128/224次完整调用，RAE分别为100/200次；部署额外弱前缀均为0。'
        '质量采样时所有控制也计算了来源诊断头，因此不能拿这些耗时声称候选无额外开销。'
        '另做同卡、同批量、3次交替重复的采样＋解码计时；原native基线移除来源头和hook。'
        '安装诊断的native输出与原native逐位一致，公式上限检查通过。\n',
        t['inference_timing'].to_markdown(index=False)+'\n',
        '\n以上仅是小规模计时，不代替大吞吐基准。离线端点、来源收集、来源头拟合成本分别在工作簿保留。'
        '暂缓的局部读出训练耗时也保留，SiT约30.24秒、RAE约67.09秒；其生成臂数为0。'
        '训练初次启动在配置校验时遇到字段类型错误，尚未训练即退出；旧请求、源代码和错误日志已归档，修复后才开始训练。\n',
        '实际输出、噪声／类别覆盖、源与checkpoint哈希、成本累加，以及缓存特征FP64重算FID由独立审计脚本核对。'
        '这里的FID复算复用同一特征，不称独立特征提取器复现。最终检查记录见'
        '[final_verification.json](data/guidance_distribution_20260912/final_verification.json)。\n',
        '## 固定样本图与可复核资产\n']
    for model in c.MODELS:
        for track in ('cfg','ig'):
            text.append(f'\n![{NAMES[model]} {track.upper()}固定前四样本](data/guidance_distribution_20260912/{model}_{track}_first4.png)\n')
    text += [
        '\n[完整工作簿](data/guidance_distribution_20260912/source_data.xlsx) · '
        '[端点数据协议](GUIDANCE_DISTRIBUTION_DATA_PROTOCOL_20260912_ZH.md) · '
        '[采样协议](SHARED_CONTAMINATION_SAMPLING_PROTOCOL_20260912_ZH.md) · '
        '[离线过滤协议](CONTAMINATION_ENDPOINT_FILTER_PROTOCOL_20260912_ZH.md) · '
        '[实施代码](../experiments/guidance_distribution_20260912/mixture.py)。\n',
        '\n本轮结果不构成具有可靠同预算收益的新方法，长期研究目标仍未完成。共同污染假设保留，'
        '后续保留“最优分布＋共同误差分布”的主要结构，允许误差成分经过简单扰动算子，并加入受约束的加性残差。'
        '本轮生成未测试这一扩展，不能用严格模型的负结果替代对扩展模型的检验。'
        '条件联合分布的识别问题仍需处理；结构化 mismatch 的推导与后续验证见'
        '[扩展分析](AG_IG_STRUCTURED_MISMATCH_20260912_ZH.md)。\n']
    REPORT.write_text(''.join(text))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tables=frames()
    for name,df in tables.items():df.to_csv(OUT/(name+'.csv'),index=False)
    sources=pd.DataFrame([
        dict(name='Frozen sampling protocol',source=str(m.PROTOCOL),sha256=c.sha(m.PROTOCOL)),
        dict(name='Endpoint protocol',source=str(c.WORK/'docs/GUIDANCE_DISTRIBUTION_DATA_PROTOCOL_20260912_ZH.md'),
             sha256=c.sha(c.WORK/'docs/GUIDANCE_DISTRIBUTION_DATA_PROTOCOL_20260912_ZH.md')),
        dict(name='Raw experiment root',source=str(m.ROOT),sha256='directory; per-request manifests'),
        dict(name='Feedback Guidance primary paper',source='https://arxiv.org/html/2506.06085v2',sha256='see archived source manifest'),
        dict(name='Prior full derivation',source='docs/AG_IG_SHARED_CONTAMINATION_DERIVATION_20260912_ZH.md',
             sha256=c.sha(c.WORK/'docs/AG_IG_SHARED_CONTAMINATION_DERIVATION_20260912_ZH.md'))])
    with pd.ExcelWriter(OUT/'source_data.xlsx',engine='openpyxl') as w:
        for name,df in tables.items():df.to_excel(w,sheet_name=name[:31],index=False)
        sources.to_excel(w,sheet_name='Sources',index=False)
        for sheet in w.book.worksheets:
            sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
            for column in sheet.columns:
                letter=column[0].column_letter
                sheet.column_dimensions[letter].width=min(65,max(14,max(len(str(v.value or '')) for v in column)+2))
    plots(tables);grids(tables);write_report(tables)
    c.atomic(OUT/'artifact_manifest.json',dict(
        files={str(p.relative_to(c.WORK)):c.sha(p) for p in sorted(OUT.iterdir()) if p.name!='artifact_manifest.json'},
        report_sha256=c.sha(REPORT),generator_sha256=c.sha(Path(__file__))))
    print('Report complete:',REPORT,flush=True)


if __name__=='__main__':main()
