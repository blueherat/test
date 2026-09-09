# Finite capacity-to-state lifting：固定四组质量对照

用户最新要求：暂停 FK，优先实验附件49271932中 lifting；允许调系数，
后期可以不用 guidance。FK保留500张，停止进程释放显存，未作完整质量判断。

## 实际算子

L=(Phi_W)^(-1) Phi_S；先 z'=z+alpha(t)*(L(z)-z)，再Strong推进同一时间段。
主路径不再额外叠加原生IG：lifting本身实现guidance。
每段最多4个原100步shift8网格步。目标Full与逆向Base均使用每细步Heun；
反向积分是连续流逆的数值近似，不是精确离散逆。主Strong推进保留细步Euler。
区间终点相同，每段重新计算两头；每段仅一次部分lifting，无同时间无限迭代。
这与旧full_prototype的“稀疏完整lifting + 原生IG”不同，但核心复合算子已试过。
不能声称算子新颖，也没有单调质量提升定理。

## 四组一次性小范围对照

- a078_constant：alpha=.78，保持到原生引导窗口结束(t<.1)。
- a078_fade：alpha=.78*clip((t-.2)/.3,0,1)，t<=.2之后只用Strong。
- a125_constant：alpha=1.25，其他同constant。
- a125_fade：alpha=1.25*clip((t-.2)/.3,0,1)，其他同fade。

alpha按每段起点计算；切段保证不越过停止引导的细步起点。
不是宣称上述系数最优，而是同时考察强度与后期继续写入的2x2对照。
不自动扩大参数扫描；四组顺序运行，每组4GPU协同1000样本。

## 协议和解释

seed202609413，B4，labels0..999，FP32state/BF16autocast/TF32，
冻结原RAEv2EMA100080、stage1和100步shift8网格。
复用原生IG FID38.264238946601836及全部样本；不重跑基线。
记录每组所有完整/前缀调用次数、用时、输入哈希、输出哈希。
constant为298Full+198prefix/图；fade为294Full+194prefix/图。
方法包含额外计算，1K只作探索；任何小幅最好值都不是突破或无偏提升证据。
若有明确提升，才针对独立5K/计算成本继续验证。用户仍要求不写论文。

实现验证仅检查采样时间方向、恒定场一阶IG关系、alpha0退化到Strong，
以及真实首批有限性/资源；不把机制测试当成生成质量准入门槛。
