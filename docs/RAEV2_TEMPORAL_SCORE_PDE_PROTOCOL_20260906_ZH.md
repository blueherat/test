# 第3轮：时间 score PDE 的目标可辨识性审计

2026-09-06。root 已选择本轮，尚未执行本协议的数值检查。问题是：能否只借助时间 score Fokker–Planck 自洽性及 RAE 线性 bridge 的精确 Gaussian 噪声端点，唯一识别真实生成目标，从而导出免调参 guidance？这是一个解析必要条件审计，不是 RAEv2 质量实验。

固定 clean Gaussian `N(μ,v)` 三个案例：真实参考 `(0,1)`、均值反例 `(1,1)`、方差反例 `(0,4)`。它们由解析推导选择，用于区分目标；不是盲测或从生成结果择优。bridge 为 `Z_t=(1−t)X+tε`，t 从0到1表示加噪。令 a=1−t、V=a²v+t²、m=aμ，score `s=−(x−m)/V`。在 `0<t<1` 上检查由相同前向算子导出的 score PDE，并验证三个案例在 t=1 都为 `−x`，而 clean 终点及其 identity-feature Gaussian FID 不同。

固定全部 rational 网格 `t={1/100,1/4,1/2,3/4,99/100}`、`x={−2,−1,0,1,2}`。以 SymPy 作全变量恒等化简，网格只作独立代入核验，不用于选参数。进一步检查解析 probability-flow 解 `x(t)=m(t)+sqrt(V(t))*z1` 与导出的 drift 一致，因此 actual law 已被完整计算，并非将估计 score 自动等同于 actual q。

附加固定见证：在真实 `(0,1)` probability-flow drift 上加 `e(t)x`，其中 `e=t(1−t)(1−2t)`，其0到1积分为0，因此生成终点仍准确；对应 implied score 的 PDE residual 非零。这用于说明消除 PDE residual 本身没有端点质量的严格排序，不声称任何具体优化器会选中错误 Gaussian 解。

准入规则：若不同 clean 分布具有同一噪声端点、同一PDE且全域零残差，则否决“只用自洽性与该端点就识别目标”的保证及据此直接训练/采样的提案。保留加入真实内部/clean边界、配对监督及适定性条件后的 PDE 正则化价值；不否定原始 FP-Diffusion。

执行只用CPU符号/解析计算，零GPU、RAEv2、decoder、辅助训练、图像及实际FID调用。输出保存请求、协议/代码身份、逐点CSV与完整summary；耗时只包含计算脚本，阅读/下载成本另记。不得把 Gaussian FID 当作 ImageNet FID。由 root 独立复核后统一登记轮次。
