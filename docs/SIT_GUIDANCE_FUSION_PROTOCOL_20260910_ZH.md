# 小 SiT：候选融合与新 5K 自动流水线

用户授权先确认 5K，再融合想法继续调优。先等待 portfolio 七配置新 5K 完成并审计，再执行本流水线。初始确认 seed=202610060/061；本轮调参使用不同的 balanced 1K seed=202610062/063；被选中的融合方案与对照最后使用第三套 balanced 5K seed=202610064/065。每类样本数分别为50/10/50。三个阶段使用相同训练模型和ADM参考。

固定184个调参配置：7种融合×12参数=84组，另有100组原生/单组件/已有ADG对照。全部在新1K上重新采样和评估，不复用旧649组的FID。所有融合采用Heun64，阶段内固定同一初始noise/labels，随机量使用原始batch noise SHA256，历史只在完成步后更新；原生Dopri5 .7对照保留旧分段。

主干、原生depth4弱头、VAE及原有拟合冻结。复合场的顺序固定：构造参考方向→可预测残差修正/通道椭球约束（如启用）→反向动量与clean正交投影（如启用）→有限clean预测的通道方差回缩（如启用）。各组件自己的结构性质未必在组合后保留；理论依据是可检验的互补假设，不是FID保证。

IG引导峰值在t<.25乘6/7，.25≤t<.5使用峰值，其后0；peak=.7保留旧.6/.7精确常数。CFG strength为额外增量，t<.75恒定，其后0。所有修正用当前真实状态，所有额外Full/prefix查询和矩阵运算计入实测成本。

单组件/原生对照：Strong1，Dopri5 IG1，Heun IG6，局部IG6，Heun ADG6，IG APG12，IG可预测残差12，IG通道椭球12，原生CFG8，CFG APG12，CFG条件割线12，CFG通道回缩12，共100。APG与条件割线参数覆盖原先最优点并向旧网格边界外扩展。ADG为已发表公式适配，避免只与较弱原生基线比较。

融合调参结束后，对IG、CFG分别取最低FID融合配置，只有它低于该来源所有非融合组件的最低FID，才进入第三套新5K。确认同时携带调优后的原生基线和最佳非融合组件；IG保留Heun/Dopri两个原生对照。否则记录负结果，不自动扩大参数网格。边界最佳点可以作为固定配置复验，但不称全局最优。

正式调参前先进行真实模型检查：复合算子的关闭组件极限必须逐元素恢复已冻结单组件；覆盖全部新增场参数的两个时间点；每个融合家族、IG-APG、ADG完整轨迹与零强度检查；原生/旧组件的代表配置复现历史首批latent。每卡正式启动再复核原生prefix、复合极限和#40旧首批。数值不匹配在质量采样前停止。

恢复规则：每个B8输出以临时文件写入，再保存含request/config/input身份的SHA256收据。重启后仅复用收据、字节哈希和输入身份完全一致的批次，并保留原计时；不完整提交移到orphan目录重算。全配置由独立commit记录绑定聚合文件。数值发散记空FID并继续，程序错误或哈希变化停止。控制器持独占锁，只管理自己的子进程；worker在父控制器退出后停止。

报告逐组CSV、每个家族最佳点、FID/sFID/IS、Full/prefix及计时。每次FID前验证完整batch覆盖和哈希；阶段结束另对选中配置及其关键对照核对原始输出、成本，并从缓存特征独立FP64重算FID/sFID。独立算术不等于重复Inception特征抽取。

入口：`python -m experiments.sit_guidance_fusion_pipeline_20260910 --pipeline`。支持 `--status` 和 `--stop-after-current`；后者在当前配置提交后停止。要显式恢复此类暂停，先移除输出根的STOP_AFTER_CURRENT文件，再运行同一入口。普通异常退出可直接再次运行入口，从有效提交继续。输出根：`/home/zhoushunyu/data/eqvae/experiments/sit_guidance_fusion_20260910`。tmux计划会话：`sit_fusion_0910`。

## 局部弱参考＋可预测残差

`ig_local_residual`；IG。

构造：`u=N_D(S-W_local-r*C_original)`。

依据：保留局部化带来的上下文差，再移除原始gap的浅层可预测分量；检验两类修正能否互补。

固定组件：`{"locality": 2.0}`；strength=(0.6, 0.7, 0.8, 0.9)；residual=(0.25, 0.5, 1.0)。

限制：C只拟合原始gap，不能视作局部化gap的正确回归；交互可能破坏已确认方向。

## 局部弱参考＋通道椭球预算

`ig_local_metric`；IG。

构造：`u=channel_metric(N_D(S-W_local),m,r)`。

依据：先形成跨空间上下文方向，再限制通道度量下的局部离群增量；方向来源与幅度约束分别控制。

固定组件：`{"locality": 2.0}`；strength=(0.6, 0.7, 0.8, 0.9)；channel_ridge=(0.1, 0.5, 2.0)。

限制：通道空间协方差不是真实误差协方差，预算可能压掉有用修正。

## 局部弱参考＋反向动量/径向投影

`ig_local_apg`；IG。

构造：`d=N_D(S-W_local); M=d+beta*Mprev; u=P_m_perp cap(M,2||D||)`。

依据：对新的上下文差方向应用因果反向动量与clean正交投影，检验径向放大是否限制局部参考收益。

固定组件：`{"locality": 2.0}`；strength=(0.6, 0.7, 0.8, 0.9)；apg_beta=(-0.75, -0.5, -0.25)。

限制：APG从CFG迁到IG未获理论保证；投影可能移除局部化带来的有效方向。

## APG＋通道方差回缩

`cfg_apg_rescale`；CFG。

构造：`u=APG(Dc); G=m+b*a*u; G=(1-r)G+r*channel_rescale(G,m)`。

依据：APG调整时间与径向方向，通道回缩控制有限clean预测的中心方差；检验二者是否互补。

固定组件：`{"apg_beta": -0.5}`；strength=(1.5, 2.0, 2.5, 3.0)；channel_rescale=(0.25, 0.5, 0.75)。

限制：回缩可能重新改变APG的径向几何；不宣称复合操作仍保留各组件的所有性质。

## 条件割线＋APG

`cfg_secant_apg`；CFG。

构造：`d=N_Dc(S(eu+r(ec-eu))-U); u=APG(d)`。

依据：先在条件embedding内选择方向，再用APG控制重复历史增益；分开条件响应与时间几何。

固定组件：`{"apg_beta": -0.5}`；strength=(1.5, 2.0, 2.5, 3.0)；condition_scale=(0.25, 0.5, 0.75)。

限制：条件embedding插值没有一般概率解释；APG可能抹掉割线方向的收益。

## 条件割线＋通道方差回缩

`cfg_secant_rescale`；CFG。

构造：`u=N_Dc(S(eu+.5(ec-eu))-U); rescale(m+b*a*u)`。

依据：改变条件响应方向后独立校准每通道有限预测方差，检验输出增益是否仍限制效果。

固定组件：`{"condition_scale": 0.5}`；strength=(1.0, 1.25, 1.5, 1.75)；channel_rescale=(0.25, 0.5, 0.75)。

限制：方向归一化与通道回缩会相互作用；需要各自原生组件曲线对照。

## 条件割线＋APG＋通道方差回缩

`cfg_secant_apg_rescale`；CFG。

构造：`d=secant_norm; u=APG(d); rescale(m+b*a*u)`。

依据：固定次序为条件方向→历史/径向控制→通道方差校准，检查三种误差控制是否有增量收益。

固定组件：`{"condition_scale": 0.5, "apg_beta": -0.5}`；strength=(1.5, 2.0, 2.5, 3.0)；channel_rescale=(0.25, 0.5, 0.75)。

限制：更多组件可能只是重复缩放；必须超过调优后的单组件及两组件曲线才能认领融合价值。
