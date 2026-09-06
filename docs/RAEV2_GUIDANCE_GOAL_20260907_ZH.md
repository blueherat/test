# RAEv2 guidance：重新启动的 3% 目标

用户于 2026-09-07 明确重启研究：在 RAEv2 上，用有理论依据的 guidance，在 1K 或 5K 采样获得 FID 相对改善至少 3%。禁止大量调参，尤其禁止几十段外推系数搜索。此前 5% 目标及四轮停止记录属于已经结束的历史，本次不改写它们。

比较基线固定为官方 EMA、DINOv3-L K7 decoder、100-step shifted Euler、官方 IG=1.78，t∈[0.1,1]。同时保留历史上略强的 t>0.5 IG 对照。新方法要优于原官方基线，不能通过更昂贵且更差的基线制造达标。额外推理成本实测；必要时加入成本匹配官方采样。类别均衡，初始噪声配对，固定官方 nanogen Inception/FID reference。所有结果保留。冻结候选后独立 seed 确认，不宣称 50K/SOTA。

## 第一方向：由 Gaussian channel 导出的随机 guidance

相关一手资料：[SiT §2.1–2.5、§3.4、附录 B/C](https://arxiv.org/html/2401.08740v1)、[DDIM](https://arxiv.org/abs/2010.02502)。这是已有 stochastic-interpolant/DDIM 机制在 RAEv2 IG 上的实验迁移，不把它宣称为全新通用采样定理。

令 a_t=1−t，z_t=a_t X+t ε。前向 Markov channel 的转移是 z_t=(a_t/a_s)z_s+独立 Gaussian，噪声方差为 t²−(a_t/a_s)²s²。给定 clean X 后的反向条件分布可直接用 Gaussian 条件公式求得：

    V(t,s) = s²[1−(a_t s/(a_s t))²]
    σ = η sqrt(V),  r = sqrt(s²−σ²)/t,  q = a_s−r a_t
    z_s = r z_t + q G(z_t,t,c) + σ ξ

G 为原官方 native BF16 IG 结果，同一个 G 同时用于 clean 和残差噪声项。η=1 为原始 Markov channel，η=0 精确还原官方 100-step Euler 的实数公式。η=0.5 是唯一预定的部分更新设置，用于检验完整刷新有限步误差过大的可能；不再扫描 η。末步 σ=0，首步条件分布具有有限极限，无额外时间窗。

理论决定设计的部分：原生通道决定 V 的整个时间依赖，两个确定性系数满足 r a_t+q=a_s 和 r²t²+σ²=s²。有限步给定正确 clean 时精确保留边缘及 Markov 条件分布；使用预测均值时并不精确，因为缺少 posterior covariance。该偏差已列入解析测试，不隐去。与旧 full-anchor relaxation 不同，本次用 G 定义 drift 和 score，避免单独向 unguided full 分布回退。

在理想平滑的内部时间区间，以同一 clean 误差 e 导出 drift/score 误差，有 δb=e/t，δs=a_t e/t²。任意扩散 d 的漂移误差为 (1+d a_t/t)e/t。Girsanov 的局部误差系数 (1+d a_t/t)²/(4d) 在 d=t/a_t 最小，正是上述 Markov channel 的扩散。这仅为固定误差的点态上界权衡，actual sampling law 随 d 变化，不构成 FID 必然下降的定理；IG 也不必是某个正确幂次分布的 Bayes denoiser。

可证伪假说：确定性递归可能累积有相关性的 guided-field 误差；理论配平的随机转移能减弱这种积累，同时保留 IG 的去噪目标。如果 1K/5K 不改善，固定迁移失败；不因 oracle 恒等式而继续扩大搜索。

## 预定实验（在读取新 FID 前固定）

- 首轮 seed=202609071，1000 样本，类别 id mod 1000，B8；4 个臂：official100、历史 piecewise100、ancestral η=1、partial η=0.5。100 次模型调用，无训练、无额外模型、每图一次 decoder。
- 初始噪声与额外噪声使用独立 SHA256 命名空间，以全局 batch 编号定位；跨 GPU 分片不改变 batch。记录每 batch 初始噪声和标签 SHA。
- η 两个候选只作为这一理论家族的固定完整/部分转移检验。首轮若有可靠方向信号，冻结更优的候选，在 seed=202609072 的 5K 与官方、历史强基线比较。没有信号则先解释失败，再决定下一机制。
- 达标分数为 1−FID(candidate)/FID(baseline)≥0.03。正式披露采样数、独立确认、成本与 FID 小样本偏差。不能用减去 real-real FID 后的相对数冒充此分数。
