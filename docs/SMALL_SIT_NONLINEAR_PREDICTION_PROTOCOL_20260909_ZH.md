# 小SiT：非线性预测组合的冻结协议

在生成本轮质量样本前固定。用户要求尝试不用线性外推；本轮直接改变去噪预测的有限组合运算，不训练新头。目标是检验有限旋转/乘法径向变化的效果，不主张新的概率目标、正确数据流形或论文创新。

## 运算与可区分性

小SiT时间约定为 `z_t=(1-t)ε+t*x`。令 `b=1-t`，Strong/Weak速度为 `S,W`，去噪预测 `m=z+b*S`、`n=z+b*W`，`d=m-n`。逐整张图展平，定义

\[
\rho=\langle m,d\rangle/\|m\|^2,\quad q=d-\rho m,\quad
\kappa=\|q\|/\|m\|.
\]

三个固定候选为

\[
G_{\rm polar}(a)=e^{a\rho}\{\cos(a\kappa)m+a\operatorname{sinc}(a\kappa)q\},
\]
\[
G_{\rm sphere}(a)=\cos(a\kappa)m+a\operatorname{sinc}(a\kappa)q,
\qquad
G_{\rm retract}(a)=\|m\|\frac{m+ad}{\|m+ad\|}.
\]

这里 sinc(u)=sin(u)/u，sinc(0)=1。将G转换回速度 `(G-z)/b`。实现通过expm1/sinc的连续商避免b接近零时相消，精确b=0使用同一连续表达式；所有a=0直接调用原生Strong分支。clean RMS≤1e-12或retraction输入RMS≤1e-12时退回Strong，记录发生频率。没有角度/指数裁剪、动量或新强度搜索。

polar的 `dG/da|0=d`，因此其一阶变化与普通IG完全一致；有限a包含径向乘法和切向旋转。sphere与retraction的一阶导数都是q，作为径向约束对照。范数分别是 `exp(aρ)||m||`、`||m||`、`||m||`。在给定m,d上，令 `K=(qmᵀ-mqᵀ)/||m||²`，则polar等于 `exp[a(ρI+K)]m`；独立CPU用稠密矩阵指数检查闭式实现。

三者仍使用同一对Strong/Weak信息，且落在二者clean预测的线性张成空间内。有限输出一般偏离仿射直线 `m+γd`；以逐状态正交残量直接核对，不能把它夸大成生成分布不可能由其他IG策略获得。球面/极坐标原点的选择是建模假说，潜变量高维不证明posterior mean位于固定球壳。

## 已知相邻工作与历史边界

[APG](https://arxiv.org/html/2410.02416v2)已讨论clean预测上平行/正交分量、缩放与动量。[ADG](https://arxiv.org/html/2506.11039v1)已有用转角替代纯线性外推的构造；其具体投影与本轮公式不同，但足以排除“首次非线性/角度guidance”表述。本轮不称复现完整APG或ADG。仓库[ADG阅读](RAEV2_GUIDANCE_READING_ANGLE_20260906_ZH.md)和[早期RAE半径实验](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md)已记录范数约束的局限；其中token范数retraction阴性，不因迁移到小SiT整图运算而改称未经探索的新原理。

## 配对质量实验

按顺序实际生成 `ig_restarted, polar_exp, sphere_exp, sphere_retraction`，每组1000图。三项候选完整公开，不只报告最低一项。原生IG在本轮重跑，用历史已核验的每rank首8个输入做精确latent预检。

全部使用原有小SiT模型、depth4弱头、MSE VAE、相同1000噪声/类别字节、B8、FP32/主干TF32设置。bank为 `lifting_wide_scale_20260909/sit_small`，是已经多次使用的探索bank，不是独立确认。Dopri5 `rtol=.001, atol=1e-6`，分段网格 `[0,.125,.25,.375,.5,1]`，额外强度 `[.6,.6,.7,.7,0]`，每次RHS只做一次主干/双头调用。自适应NFE和逐查询几何运算可能改变实际成本；报告轨迹加解码batch GPU秒及Full/图，记录诊断成本，不能只凭相同网络调用结构称等成本。

预检必须通过：Strong/Weak对与单独完整/前缀输出精确一致、零强度四路一致、原生IG首8 latent逐位复现、FP32公式对FP64检查、矩阵指数独立检查、范数恒等式、退化输入、有限强度偏离原仿射线。实现故障不作为方法阴性；若冻结后修复，保留旧请求与明确修订链。非有限状态或Dopri5步长下溢则保留失败配置，不悄悄降低强度。

ADM评测沿用5000真实图参考和同一Inception图。全部batch必须覆盖恰好1000索引并核验noise/label/source/asset SHA；用缓存特征独立FP64复算FID及sFID，报告IS。固定最前8张作可视核查。FID-1K只用于本轮探索；若候选优于原生IG，再冻结选定候选、另建新噪声做5K确认，不能把已看过的5K银行当新确认，也不由1K直接宣布成功。原论文研究目标仍未完成。
