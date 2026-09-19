"""Render source-backed Chinese research tables from the completed audit bundle."""
from pathlib import Path
import json
from experiments.guidance_pasted_20260912 import common as c

OUT=c.WORK/'docs/data/weak_reference_20260914'
TARGET=c.WORK/'docs/WEAK_REFERENCE_GUIDANCE_RESEARCH_20260914_ZH.md'


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',
        *('|'+'|'.join(str(x) for x in row)+'|' for row in rows)])


def main():
    validation=c.read(OUT/'bundle_validation.json');assert validation['complete']
    data=c.read(OUT/'all_image_results.json');assert data['complete']
    index={(r['phase'],r['model'],r['arm']):r for r in data['rows']}
    def get(phase,arm,model='sit'):return index[phase,model,arm]
    def f(row):return f"{row['fid']:.3f}"
    labels={'strong_e320':'Euler320','strong_band_w1':'双尺度 .2/.4，强度 1，64 步',
        'strong_log_k04_e64_w1':'单尺度 κ=.4，ω=1，64 步',
        'strong_log_k04_e106_w1':'单尺度 κ=.4，ω=1，106 步',
        'strong_log_k04_e106_w075':'单尺度 κ=.4，ω=.75，106 步',
        'strong_e64':'Euler64','strong_e192':'Euler192'}
    def standard(phase,arms):
        return table(['配置','FID↓','前向/图','采样及解码秒/1K'],
            [[labels.get(a,a),f(r:=get(phase,a)),r['full_calls_per_output'],f"{r['seconds']:.1f}"] for a in arms])
    cal=c.read(OUT/'calibration/results.json');calstrong=c.read(OUT/'calibration_strong/results.json')
    for a,b in zip(cal['rows'],calstrong['rows']):
        assert a['weights']==b['weights'] and abs(a['delta']-b['delta'])<1e-15
    calibration=(
        '真实 SiT 的三个核都在前七个时间段选择了 0；只有最后一个时间段的系数非零。'
        '保留 CFG 与去掉 CFG 的两次校准得到相同权重和验证损失差；基线速度 MSE 分别为 '
        f"{cal['rows'][0]['baseline_validation_mse']:.9f} 与 {calstrong['rows'][0]['baseline_validation_mse']:.9f}。\n\n"+
        table(['固定核','最后时间段系数','验证 MSE 变化','按图像聚类的标准误'],
            [[r['kernel'],f"{r['weights'][-1]:.6f}",f"{r['delta']:+.3e}",f"{r['cluster_image_se']:.3e}"] for r in cal['rows']])+
        '\n\n三个变化均略高于零且接近标准误，不能认定改善。原始 `selected_kernel=lowpass` '
        '仅表示候选核中损失最低，未包括基线；补充的接受门槛将基线纳入后，保留基线。'
        '因此没有部署这些 DSM 拟合权重的图像采样。另行运行的单尺度与双尺度固定强度实验不使用该选择结果。')
    images=[]
    images.append('每个配置生成 1,000 张图像。筛选种子为 `2026091407`，单尺度独立复核为 '
        '`2026091408`；SG 与双尺度补充实验沿用该复核输入。双尺度的独立对照使用 '
        '`2026091409`。合计 34 个配置、34,000 张生成图像；另有两种基线下的真实加噪数据校准，'
        '这些加噪状态不计入生成图像数。')
    images.append('先看统一单尺度参数 `κ=0.2, ω=1`，保留各模型原有 CFG/IG 设置：\n\n'+
        table(['设置','原基线 FID','加 log 外推','同前向次数 Euler','log / 基线实测时间'],[
            [label,f(b:=get(phase,base,model)),f(g:=get(phase,guided,model)),f(get(phase,budget,model)),f"{g['seconds']/b['seconds']:.2f}×"]
            for label,phase,model,base,guided,budget in [
                ('SiT，无 CFG，独立复核','strong_confirm_1k','sit','strong_e64','strong_log_k02_w1','strong_e192'),
                ('SiT，CFG','sit_screen_1k','sit','cfg_e64','cfg_log_k02_w1','cfg_e137'),
                ('JiT，CFG','cross_screen_1k','jit','cfg','cfg_log_k02_w1','cfg_equal_nfe'),
                ('RAEv2，IG','cross_screen_1k','raev2','ig','ig_log_k02_w1','ig_equal_nfe')]]))
    pilot=get('sit_screen_1k','strong_e64');pilotlog=get('sit_screen_1k','strong_log_k02_w1')
    images.append(f"SiT 不加 CFG 的最初筛选也从 {f(pilot)} 降至 {f(pilotlog)}，方向在独立种子上重复。"
        '这比仅报告加 CFG 后的微小差异更有价值，但其绝对质量仍明显低于本模型的 CFG 基线，不能宣称替代 CFG。')
    boot=c.read(OUT/'bootstrap/results.json')
    images.append('对独立复核样本进行 200 次按类别分层的配对 bootstrap，固定真实参考统计量，结果如下。'
        '区间只描述当前 1K 样本的条件不确定性，不消除 FID 小样本偏差。\n\n'+
        table(['单尺度相对对照','FID 差值','bootstrap 95% 分位区间'],[
            [labels.get(r['contrast'].split(' minus ')[1]),f"{r['observed_delta']:+.3f}",f"[{r['percentile_95_interval'][0]:+.3f}, {r['percentile_95_interval'][1]:+.3f}]"] for r in boot['rows']]))
    images.append('在同一组独立复核输入上补上原 SG 与双尺度对照：\n\n'+
        table(['方法','FID↓','前向/图','采样及解码秒/1K'],[
            [label,f(r:=get(phase,arm)),r['full_calls_per_output'],f"{r['seconds']:.1f}"]
            for label,phase,arm in [
                ('原模型 Euler64','strong_confirm_1k','strong_e64'),
                ('原模型 Euler192','strong_confirm_1k','strong_e192'),
                ('单尺度 κ=.2, ω=1','strong_confirm_1k','strong_log_k02_w1'),
                ('原 SG，64 步，δ=.01, ω=1','strong_sg_1k','strong_sg_e64_w1'),
                ('原 SG，96 步，δ=.01, ω=1','strong_sg_1k','strong_sg_e96_w1'),
                ('双尺度 .2/.4，强度 1','strong_band_1k','strong_band_w1'),
                ('原模型 Euler320','strong_band_1k','strong_e320')]]))
    images.append('SG 的 96 步版本为 191 次前向，接近单尺度的 192 次。上述比较只针对给定 SG 参数。'
        '双尺度在这一轮比单尺度更好，但其预算和外层扰动都更大，因此追加独立种子的消融。'
        '其中单尺度强度 0.75 满足 `(0.4²−0.2²)/0.4²=0.75`，匹配双尺度的小扰动一阶系数；'
        '106 步单尺度为 318 次前向，与双尺度的 320 次相差不足 1%。\n\n'+
        standard('strong_band_confirm_1k',['strong_e320','strong_band_w1','strong_log_k04_e64_w1',
            'strong_log_k04_e106_w1','strong_log_k04_e106_w075']))
    bandboot=c.read(OUT/'bootstrap_band/results.json')
    images.append('双尺度独立消融的配对 bootstrap：\n\n'+table(['双尺度相对对照','FID 差值','bootstrap 95% 分位区间'],[
        [labels.get(r['contrast'].split(' minus ')[1]),f"{r['observed_delta']:+.3f}",f"[{r['percentile_95_interval'][0]:+.3f}, {r['percentile_95_interval'][1]:+.3f}]"]
        for r in bandboot['rows']]))
    images.append('CFG 条件下其余方案与超参数的完整筛选如下，均为种子 `2026091407`。'
        '基线在不同阶段重复计算的 FID 有约 0.0005 的特征计算浮点差异，比较使用阶段内基线。\n\n'+
        table(['配置','FID↓','前向/图'],[
            [label,f(r:=get(phase,arm)),r['full_calls_per_output']]
            for label,phase,arm in [
                ('CFG Euler64','sit_screen_1k','cfg_e64'),
                ('log κ=.1, ω=1','sit_screen_1k','cfg_log_k01_w1'),
                ('log κ=.2, ω=1','sit_screen_1k','cfg_log_k02_w1'),
                ('log κ=.2, ω=3','sit_screen_1k','cfg_log_k02_w3'),
                ('log κ=.4, ω=1','sit_screen_1k','cfg_log_k04_w1'),
                ('原 SG δ=.01, ω=1','sit_screen_1k','cfg_sg_w1'),
                ('旋转初始角 .1','angular_screen_1k','cfg_angular_a01_w1'),
                ('旋转初始角 .2','angular_screen_1k','cfg_angular_a02_w1'),
                ('矩匹配 κ=.2, ω=1','moment_screen_1k','cfg_moment_k02_w1'),
                ('CFG Euler100','moment_screen_1k','cfg_e100'),
                ('CFG Euler137','sit_screen_1k','cfg_e137')]]))
    images.append('矩匹配未超过其 175 次前向的 Euler100 对照；旋转及 log 筛选未超过 240 次前向的 Euler137。'
        '这些结论限定在当前 CFG 和参数设置。原 SG 在本次 CFG 筛选较好，但其此前独立复核并不稳定，'
        '可参阅 [先前 SG 实验](SELF_GUIDANCE_SIT_20260913_ZH.md) 与 '
        '[先前 JiT/RAEv2 SG 实验](SELF_GUIDANCE_JIT_RAEV2_20260913_ZH.md)。')
    images.append('固定样本对照图：\n\n'
        '![无 CFG 的单尺度独立复核，预先固定的前六个样本](data/weak_reference_20260914/strong_confirm_1k/sit/comparison.png)\n\n'
        '[双尺度独立消融图](data/weak_reference_20260914/strong_band_confirm_1k/sit/comparison.png)、'
        '[JiT 对照图](data/weak_reference_20260914/cross_screen_1k/jit/comparison.png)、'
        '[RAEv2 对照图](data/weak_reference_20260914/cross_screen_1k/raev2/comparison.png)。')
    decisions=c.read(Path(__file__).with_name('research_verdict.json'))
    template=Path(__file__).with_name('research_template.md').read_text()
    for key,value in dict(TOPLINE=decisions['topline'],CALIBRATION=calibration,
                          IMAGE_RESULTS='\n\n'.join(images),DECISIONS=decisions['decisions']).items():
        template=template.replace('{{'+key+'}}',value)
    assert '{{' not in template
    TARGET.write_text(template)
    print(str(TARGET),len(template),'characters')


if __name__=='__main__':main()
