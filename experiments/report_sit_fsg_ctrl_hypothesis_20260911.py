"""Write final Chinese research readouts from complete, audited experiment tables."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from experiments.lifting_scale_sweep_20260909 import WORK,read,sha
from experiments.sit_fsg_ctrl_hypothesis_20260911 import pipeline as p
from experiments import sit_golden_path_hypothesis_20260911 as direct
from experiments.analyze_sit_fsg_ctrl_hypothesis_20260911 import NAMES

DIRECT=WORK/'docs/data/sit_golden_path_direct_20260911'
XL=WORK/'docs/data/sit_xl_fsg_ctrl_examples_20260911'


def csv(path):
    # The actual method/suffix name "null" must survive CSV loading.
    return pd.read_csv(path,keep_default_na=False,na_values=[''])


def pct(value):return f'{100*value:.2f}%'


def direct_report():
    coverage=read(DIRECT/'coverage.json');audit=read(DIRECT/'data_audit.json')
    assert coverage['complete'] and coverage['rows']==1344 and audit['passed'] and audit['complete']
    frame=csv(DIRECT/'rows.csv');traces=csv(DIRECT/'optimization_traces.csv')
    core=frame[frame.variant=='optimized']
    last=core[core.iteration==64];initial=core[core.iteration==0]
    agreement=last[last.objective=='agreement']
    means=core.groupby(['k','objective','iteration']).mean(numeric_only=True)
    final=last.groupby('objective').mean(numeric_only=True)
    labels=dict(agreement='同状态未来一致性',write='冻结原条件终点',local='当前速度相等')
    lines=['# 直接检验“无条件 = 有条件”：完整图像实验\n',
        '**图像出现了“更一致，但更差”的实例。** 本次在完整latent上直接优化三个不同的相等目标，固定16个噪声、三个时刻，所有预定实验均已完成。部分NULL结局的类别读出提高；同时存在两条未来接近、但共同偏离原条件图像并丢失结构的实例。不能把分类器概率提高或两条图像相似单独当成质量提高。\n',
        '这里检验的是显式定义的损失，与FSG SiT适配器的质量比较分开。作者代码并没有直接求解下面的全程一致性损失；其执行情况见[作者代码核对](SIT_FSG_AUTHOR_CODE_AUDIT_20260911_ZH.md)。\n',
        '记捕获的状态为x₀，U(x)、C(x)分别表示从当前时刻开始、纯NULL和纯条件推进到终点；G(x)表示继续原CFG。比较以下目标，所有模型权重冻结，分类器不参与优化：\n',
        '|目标|优化损失|条件终点是否可以随状态变化|',
        '|---|---|---|',
        '|同状态未来一致性|‖U(x)−C(x)‖²|可以，两个分支都求导|',
        '|冻结原条件终点|‖U(x)−C(x₀)‖²|原目标固定|',
        '|当前速度相等|‖v_c(x)−v_u(x)‖²|只比较当前预测|',
        '\nSiT-S/2、ImageNet-100、256px；时间从噪声0到图像1，真NULL为空类别100。固定原机制集前16个噪声与CFG a=1.25轨迹，时刻t=0.125/0.375/0.625；CFG在t≥0.75后使用纯条件分支。完整4096维状态使用64次L-BFGS外层迭代，每批2个状态；损失按逐样本初始MSE归一化，并逐样本回退增大的更新。优化使用12个剩余Heun区间，最终读数重新运行原64网格剩余段，表中报告RMS比值而非MSE比值。每次同时保存NULL、纯条件和CFG终点。\n',
        '下表只列“同状态未来一致性”目标。每行均为同一组16个噪声，前→后表示原状态与64次优化后的配对均值；类别top1来自后验ConvNeXt，仅是类别对齐读出。\n',
        '|时刻|细网格U/C差 / 初值|NULL top1 前→后|条件 top1 前→后|CFG top1 前→后|条件终点漂移 / 原U/C差|位移RMS / 半径倍数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k in direct.TIMES:
        before=means.loc[(k,'agreement',0)];after=means.loc[(k,'agreement',64)]
        pairs=[f'{pct(before[x+"_convnext_top1"])} → {pct(after[x+"_convnext_top1"])}' for x in ('u','c','g')]
        lines.append(f'|{k/64:g}|{after.agreement_ratio:.3f}|'+ '|'.join(pairs)+
            f'|{after.conditional_drift_over_initial_gap:.2f}|{after.shift_rms:.3f} / {after.shift_over_reference_radius:.1f}×|')
    lines.extend([
        f'\n48个状态的最终细网格U/C差与初值之比：均值 {agreement.agreement_ratio.mean():.3f}，'
        f'中位数 {agreement.agreement_ratio.median():.3f}，范围 {agreement.agreement_ratio.min():.3f}–{agreement.agreement_ratio.max():.3f}。'
        f'其中 {int((agreement.agreement_ratio<.1).sum())}/48 低于0.1。残差明显降低，但并非严格为零。\n',
        '三种目标的最终结果如下。这里先在每个状态上归一化，再对16噪声×3时刻取均值，避免把初始差异当成改善：\n',
        '|优化目标|细U/C差 / 初值|对原条件目标的误差 / 初值|当前速度差 / 初值|状态位移RMS|',
        '|---|---:|---:|---:|---:|'])
    for objective in direct.OBJECTIVES:
        row=final.loc[objective]
        lines.append(f'|{labels[objective]}|{row.agreement_ratio:.3f}|{row.write_ratio:.3f}|{row.local_gap_ratio:.3f}|{row.shift_rms:.3f}|')
    lines.extend([
        '\n这三个目标的导数也不同：未归一化平方损失的梯度分别为 2(J_U−J_C)ᵀ(U−C) 与 2J_Uᵀ(U−C(x₀))。前者允许条件与NULL一起移动，后者保留要转移的原终点。直接数据因此可以区分“共同未来变得接近”和“原条件结果被NULL复现”。\n',
        '以下图片固定展示索引0–3，未按结果筛选。最早时刻的键盘样本 #001 是清楚的负例：优化前CFG能看到键盘结构，优化后NULL、条件、CFG三者共同变成缺少键盘结构的褐色图像。中间时刻同一键盘样本仍可识别，但字符与边缘细节改变；Chihuahua样本则进一步丢失面部结构。类别概率与人眼结构判断有时分歧，因此保留两套分类器和全部原始图像。\n',
        '![同状态未来一致性：最早时刻固定前4张](data/sit_golden_path_direct_20260911/examples_agreement_08_64.png)\n',
        '![同状态未来一致性：中间时刻固定前4张](data/sit_golden_path_direct_20260911/examples_agreement_24_64.png)\n',
        '![冻结原条件终点：中间时刻固定前4张](data/sit_golden_path_direct_20260911/examples_write_24_64.png)\n',
        '![各时刻残差随优化变化](data/sit_golden_path_direct_20260911/fine_future_agreement.png)\n',
        '![各时刻三种续生成的类别读出](data/sit_golden_path_direct_20260911/target_readouts_by_time.png)\n',
        '大位移是本次解释的实际限制。晚时刻原始U/C差和参考半径较小，归一化漂移的分母也较小，需要同时看绝对位移与图片。参考半径r=(4/64)a‖g‖；下表将一致性优化最终位移投影到r或4r，并加入与其完整位移同长度的CFG方向。投影后的状态均重新生成和测量；投影不是约束域内重新优化，因此不能用于断言“所有小位移解均不存在”。\n',
        '|一致性优化后的对照|细U/C差 / 初值|条件终点漂移 / 原U/C差|CFG目标概率|位移RMS|',
        '|---|---:|---:|---:|---:|'])
    controls=frame[(frame.objective=='agreement')&(frame.iteration==64)].groupby('variant').mean(numeric_only=True)
    for variant,title in [('optimized','完整优化位移'),('radius_projected','投影到r'),('radius4_projected','投影到4r'),('cfg_equal','同长度CFG方向')]:
        row=controls.loc[variant]
        lines.append(f'|{title}|{row.agreement_ratio:.3f}|{row.conditional_drift_over_initial_gap:.2f}|{row.g_convnext_p:.3f}|{row.shift_rms:.3f}|')
    checks=csv(DIRECT/'gradient_checks.csv')
    maximum=checks.scalar_loss_autograd_vs_finite_relative_error.max()
    lines.extend([
        '\n本实验没有验证状态仍位于论文要求的概率流形，也没有达到精确零残差。因此，结果不否定附带流形约束的严格等式假设；它不支持把“一致性损失下降”直接当作图像质量提高的证据。16张不计算FID，也不据此作总体质量排名。原1K实验中的一次4维拟合与这里的完整latent优化也不是同一个方法。\n',
        f'梯度检查的最大AD/中心差分相对偏差为 {100*maximum:.4f}%；可微12步求解器与普通12步求解器逐位相同。'
        f'全部状态数组和请求哈希通过检查，保存状态重算位移RMS最大偏差 {audit["max_saved_state_rms_error"]:.3g}，'
        '同长度对照、投影半径、各目标原始状态相同及逐样本粗损失不增加均已核验。这些检查没有把粗网格下降替代为细网格结果。\n',
        f'[全部16样本与迭代图册]({direct.ROOT}/gallery.html) · '
        '[逐状态结果](data/sit_golden_path_direct_20260911/rows.csv) · '
        '[完整优化轨迹](data/sit_golden_path_direct_20260911/optimization_traces.csv) · '
        '[数据核验](data/sit_golden_path_direct_20260911/data_audit.json) · '
        '[固定协议](SIT_GOLDEN_PATH_DIRECT_PROTOCOL_20260911_ZH.md)。\n',
        f'实验请求SHA256：`{sha(direct.ROOT/"request.json")}`。\n'])
    path=WORK/'docs/SIT_GOLDEN_PATH_DIRECT_RESULTS_20260911_ZH.md'
    path.write_text('\n'.join(lines));return path


def main_report():
    coverage=read(p.PORTABLE/'coverage.json')
    assert coverage['quality_complete'] and all(coverage['complete'].values())
    quality=csv(p.PORTABLE/'quality_all_results.csv');assert len(quality)==36
    assert quality[['resnet_top1','convnext_top1']].notna().all().all()
    quality_audit=read(p.ROOT/p.STAGE/'analysis_audit.json');assert quality_audit['passed']
    trajectory=csv(p.PORTABLE/'trajectory_means.csv')
    interventions=csv(p.PORTABLE/'intervention_rows.csv')
    full=quality[quality.handoff.isna()].set_index('method')
    cfg=full.loc['cfg_tuned'];ctrl=full.loc['smc_high'];fsg=full.loc['fsg_high']
    lines=['# FSG / CFG-Ctrl：图像机制与配对1K结果\n',
        '**本轮SiT适配里，FSG在早期撤条件时有收益，完整轨迹的FID却更差；CTRL明显优于同参数的瞬时修正。** '
        f'调参后的原生CFG FID为 {cfg.fid:.4f}，CTRL为 {ctrl.fid:.4f}，FSG适配为 {fsg.fid:.4f}。'
        '完整latent直接一致性实验也已完成：两条未来接近可以伴随共同退化，具体残差、图像和位移限制见[直接相等检验](SIT_GOLDEN_PATH_DIRECT_RESULTS_20260911_ZH.md)。\n',
        '**实现范围需明确：当前FSG是SiT上的算子适配，不能称完整复现作者SDXL pipeline。** '
        '作者代码的普通步执行CFG++，使用校准前缓存的NULL预测；本轮使用Heun CFG，并有预先固定的半径裁剪和不同的前瞻调度。'
        '已检查并执行作者真实函数体，调度器时间系数还存在需要单独区分的实际步距问题。代码版本、逐行依据、CPU重放与适用边界见[作者代码核对](SIT_FSG_AUTHOR_CODE_AUDIT_20260911_ZH.md)。\n',
        '实验使用SiT-S/2 ImageNet-100 800K checkpoint、原VAE、256px、Heun64。时间从噪声0到图像1；'
        'CFG为v_c+a(v_c−v_u)，对应常见权重w=1+a。t≥0.75关闭附加guidance后仍使用条件分支；真正NULL后缀使用空类别100。'
        '机制集为全新200个噪声、100类各2张；1K质量集为同一套配对旧噪声、100类各10张，参考为固定5000张验证集的ADM Inception统计。'
        'a=1.25和CTRL参数来自此前1K选择，因此这些质量结果是机制探索，不是独立确认。\n',
        '新增32项1K采样全部完成，另4条基线在首批latent与调用数逐位复现后复用。完整轨迹质量如下；分类器读取计算FID的同一份uint8像素，CN为ConvNeXt-Tiny，R18为ResNet18，top1仅衡量类别对齐。Full为每张轨迹的完整模型求值次数，耗时为采样加解码的累计batch GPU秒，倍数相对于调参后CFG。\n',
        '|方法|FID↓|sFID↓|CN目标top1|Full|耗时倍数|同目标类Inception方差比|',
        '|---|---:|---:|---:|---:|---:|---:|']
    order=['cfg_tuned','cfg_high','fsg_high','fsg_length_high','fsg_debiased_high','cycle_high',
        'refined_high','smc_high','instant_high','instant01_high','soft_high','norm_high',
        'local_fit','agreement_fit','write_fit','guided_write_fit']
    for method in order:
        row=full.loc[method]
        lines.append(f'|{NAMES[method]}|{row.fid:.4f}|{row.sfid:.3f}|{pct(row.convnext_top1)}|'
            f'{row.full_calls_per_image:g}|{row.sampling_cost_ratio_tuned_cfg:.2f}×|{row.within_target_variance_ratio_tuned_cfg:.3f}|')
    lines.extend([
        f'\n把每次计算出的FSG位移长度改沿局部CFG方向施加，FID为 {full.loc["fsg_length_high"].fid:.4f}，'
        f'好于FSG适配的 {fsg.fid:.4f}，但仍差于不加校准的强CFG {full.loc["cfg_high"].fid:.4f}。'
        f'同为260次Full的Heun加密为 {full.loc["refined_high"].fid:.4f}。'
        '这些对照说明，方向变化和额外计算本身没有在本轮转化成质量收益；位移幅度、普通采样器与往返离散误差都需要分别评估。\n',
        '\n四种fit是在4维子空间中、t=0.375做一次更新的预定方法，均需1005次Full；它们的FID接近原CFG，微小差值不能解释成可靠质量收益。'
        '这组结果不能代替后续完整4096维优化实验。FSG的同目标类Inception方差更小，是分布收缩的诊断线索；它不是经过验证的多样性指标。\n',
        '撤除全部条件后，FSG适配的类别保持较早出现。下面是200个新噪声的ConvNeXt目标top1；后续NULL为真正无条件推进，不能与关闭额外CFG混称。\n',
        '|前缀方法|t=0.25转NULL|t=0.5转NULL|t=0.75转NULL|继续原guidance|',
        '|---|---:|---:|---:|---:|'])
    for method in ['cfg_tuned','cfg_high','fsg_high','fsg_length_high','smc_high','instant_high']:
        values=[]
        for k,suffix in [(16,'null'),(32,'null'),(48,'null'),(64,'guided')]:
            row=trajectory[(trajectory.method==method)&(trajectory.k==k)&(trajectory.suffix==suffix)].iloc[0]
            values.append(pct(row.convnext_top1))
        lines.append('|'+NAMES[method]+'|'+'|'.join(values)+'|')
    lines.extend([
        '\n类别保持的优势并不在所有切换时刻对应FID优势。以下是每组1000图的后缀FID，均从对应前缀继续生成：\n',
        '|前缀方法|t=0.25转NULL|t=0.5转NULL|t=0.75转NULL|t=0.5转纯条件|',
        '|---|---:|---:|---:|---:|'])
    for method in p.HANDOFF_METHODS:
        values=[]
        for k,tail in [(16,'null'),(32,'null'),(48,'null'),(32,'conditional')]:
            row=quality[(quality.method==method)&(quality.handoff==k)&(quality['tail']==tail)].iloc[0]
            values.append(f'{row.fid:.4f}')
        lines.append('|'+NAMES[method]+'|'+'|'.join(values)+'|')
    def suffix_value(method,k,metric='fid'):
        return quality[(quality.method==method)&(quality.handoff==k)&(quality['tail']=='null')].iloc[0][metric]
    lines.extend([
        f'\n在t=0.25转NULL时，FSG适配的FID {suffix_value("fsg_high",16):.4f} '
        f'优于同强度CFG的 {suffix_value("cfg_high",16):.4f}；t=0.5时则为 '
        f'{suffix_value("fsg_high",32):.4f} 对 {suffix_value("cfg_high",32):.4f}，排序翻转。'
        '因此，早期撤条件实验支持FSG适配较早保留条件内容，但不能把它当成在所有时刻都改善质量的机制。'
        '这一1K后缀组没有FSG长度沿CFG方向的完整质量对照，尚不能把早期FID优势单独归因于FSG的方向变化。\n',
        '三个NULL切换时刻，CFG/CTRL的Full次数分别为 '+
        '/'.join(f'{suffix_value("cfg_tuned",k,"full_calls_per_image"):g}' for k in [16,32,48])+
        '，FSG适配为 '+
        '/'.join(f'{suffix_value("fsg_high",k,"full_calls_per_image"):g}' for k in [16,32,48])+
        '。较早撤条件减少模型查询，也改变质量；各配置的实际时间和类别读出保留在完整质量表。\n',
        '![配对1K：FID与目标类别保持](data/sit_fsg_ctrl_hypothesis_20260911/figures/fid_and_alignment_handoff.png)\n',
        '\n![固定200图的NULL接管读出](data/sit_fsg_ctrl_hypothesis_20260911/figures/null_handoff_alignment.png)\n',
        '![固定前4张完整轨迹](data/sit_fsg_ctrl_hypothesis_20260911/figures/examples_guided_64.png)\n',
        '同状态、同位移半径的干预进一步区分三种目标。下表对64个噪声、两档强度、三个时刻取配对均值，误差下降正值表示改善：\n',
        '|更新方向|对原条件终点的误差下降|修改后U/C差下降|NULL目标top1变化|',
        '|---|---:|---:|---:|'])
    ig=interventions.groupby('variant').mean(numeric_only=True)
    for variant,title in [('cfg_radius','CFG方向'),('fsg_radius','FSG方向'),('local_radius','局部gap负梯度'),
        ('agreement_radius','未来一致性负梯度'),('write_radius','冻结条件终点负梯度')]:
        row=ig.loc[variant]
        lines.append(f'|{title}|{pct(row.reduction_write_rms)}|{pct(row.reduction_agreement_rms)}|{100*row.delta_u_convnext_top1:+.2f} pp|')
    oracle=ig.loc['full_inverse_oracle']
    lines.extend([
        f'\n前32个噪声的完整NULL反演oracle，把对原条件终点的误差降低 {pct(oracle.reduction_write_rms)}，'
        f'但修改后U/C差只降低 {pct(oracle.reduction_agreement_rms)}，局部速度差反而增加 {pct(-oracle.reduction_local_gap_rms)}。'
        '它把冻结的原条件结果转移给NULL，并未让新状态的两条未来相等。oracle未受同一半径约束，不能把它的改善与小步对照直接作效率比较。\n',
        'Jacobian实现已用独立自动微分和中心差分核对；原先“同长度CFG更接近”的解释必须加上目标定义。'
        'H=1/32时，以精细guided流终点为目标，Euler FSG误差26.82%，同长度CFG为18.28%；'
        '以FSG自己的Euler前向终点为目标，FSG为13.88%，同长度CFG为15.76%。'
        '未裁剪FSG对自己的离散目标更有优势，裁剪可能改变这一结果。这里比较的是直接终点误差，无需假定Jacobian线性化成立。\n',
        '严格的逆流组合δ=Φᵤ⁻¹(Φw(x))−x，可以在局部写成 Jᵤ⁻¹[Φw(x)−Φᵤ(x)]；'
        '它作用于一个具体未来位移，并没有显式估计完整Jacobian矩阵。H→0时δ≈H(1+a)g。'
        '同长度CFG对照取‖δ‖g/‖g‖，不是另选一个更强超参数。数值检查、有限步余项和目标纠正见[Jacobian复核](SIT_FSG_JACOBIAN_AUDIT_RESULTS_20260911_ZH.md)。\n',
        'CTRL的历史项有可见作用。公共离散形式为 mₖ=gₖ−K sign[gₖ+(λ−1)mₖ₋₁]，本轮K=0.2、λ=5。'
        '固定小正gap 0<g<0.16时，m在g−K和g+K之间形成二周期，平均值为g；瞬时式g−K sign(g)则持续反转这个小gap。'
        '实际轨迹第1步，历史项使约68.17%的坐标修正符号不同于sign(g)，并非罕见扰动。'
        '本轮Heun两阶段共用冻结历史，并提交左阶段的修改gap；这些离散证据不能直接当成连续时间导数控制的证明。\n',
        '![CTRL实际逐坐标读出](data/sit_fsg_ctrl_hypothesis_20260911/figures/ctrl_coordinate_traces.png)\n',
        'SiT-XL/2也已完成固定16噪声、12方法、5种续生成，共960张图片。使用官方SiT-XL-IG800EP checkpoint的full分支、NULL=1000，'
        '时间与速度映射通过原生Euler逐位核对。该checkpoint经过IG训练，不能等同于未训练IG的原始SiT模型；参数沿用小模型探针，未在16张上重新调参。'
        '这组单图用于检查现象是否可见，不提供大模型FID或普遍排名。\n',
        '![SiT-XL固定前4张](data/sit_xl_fsg_ctrl_examples_20260911/main_guided_64.png)\n',
        f'[小模型全部200样本图册]({p.ROOT}/gallery.html) · '
        '[大模型全部16样本图册](/home/zhoushunyu/data/eqvae/experiments/sit_xl_fsg_ctrl_examples_20260911/gallery.html) · '
        '[完整36项质量表](data/sit_fsg_ctrl_hypothesis_20260911/quality_all_results.csv) · '
        '[直接一致性实验](SIT_GOLDEN_PATH_DIRECT_RESULTS_20260911_ZH.md)。\n',
        f'32项新1K的请求、源文件、权重、输入身份、样本覆盖与全部元数据通过核验；其中 {len(quality_audit["raw_and_metric_audits"])} 项另逐批重算原始文件哈希、核对实际调用与计时，并从缓存特征独立重算FID/sFID，绝对偏差均小于0.001。'
        '没有重新提取Inception特征。分类器另从相同FID像素读取，并记录图像/权重哈希。'
        f'[核验记录]({p.ROOT/p.STAGE}/analysis_audit.json) · [固定协议](SIT_FSG_CTRL_HYPOTHESIS_PROTOCOL_20260911_ZH.md)。\n',
        f'机制请求SHA256：`{sha(p.ROOT/"study_request.json")}`；质量请求SHA256：`{sha(p.ROOT/p.STAGE/"request.json")}`。\n'])
    path=WORK/'docs/SIT_FSG_CTRL_HYPOTHESIS_RESULTS_20260911_ZH.md'
    path.write_text('\n'.join(lines));return path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--direct-only',action='store_true')
    args=parser.parse_args();print(direct_report())
    if not args.direct_only:print(main_report())
