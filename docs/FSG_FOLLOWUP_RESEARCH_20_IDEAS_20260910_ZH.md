# FSG 后续研究与二十项可检验假设

FSG 的条件承载观点值得继续检验。最有价值的延伸是明确“写入成功”还需要满足什么：保留原本条件结局、跨时间与扰动仍可读取，或在均值之外识别不确定性、竞争类别和重复写入。本轮将这些问题落实为 **8 项围绕原等式的候选、12 项改变信息判据的候选**，各做 10 组小 SiT paired 1K。候选使用相同基础采样器、输入和评测；APG 等已发表算法独立作为对照。

这些是有明确动机和可执行定义的研究候选，不声称已经得到二十项论文级原创方法。若两项共享数学结构，会直接说明共享部分。FID结果、机制成立与公开文献新颖性是三个需要分别验证的问题。

## FSG 的出发点与本轮边界

FSG 研究的是从同一中间状态出发，条件与无条件**续生成映射**的结局一致。论文以有限区间和固定点迭代近似这一目标，并在特定时刻进行前向、反向校准；常规步骤仍使用 CFG++。不能把完整续生成等式直接替换为“任何时刻两个单步速度相等”，也不能把原算法描述为只写入一次后永久撤条件。[^fsg]

仓库已有的 [条件接管实验](FSG_CONDITION_HANDOFF_RESULTS_20260908_ZH.md) 支持这个研究动机：RAE 在高噪状态写入后，后续仅运行 null 分支仍保留了明显的目标类别信息。但这组小规模机制实验不能替代当前 SiT 的质量和成本比较。[单边逆映射](FSG_ANCHORED_INVERSE_PROBE_20260908_ZH.md)、[多步求解](FSG_ANCHORED_INVERSE_MULTISTEP_20260908_ZH.md) 和 [同场往返对照](FSG_COMMON_FIELD_CONTROL_PROTOCOL_20260908_ZH.md) 也已经做过，因此不把换一个求解器或再次执行原算子计作新候选。

本轮保留一个具体检验方向：信息确实可能被写入 latent，而“成功写入”的操作性判据仍可以改进。对于完美模型，继续增加 guidance 也不自动优于正确条件分布；本轮所有质量预测都针对有限模型和实际数值采样中的误差。

## 可核验的引用关系

检索截止到 2026-09-10。以 FSG 的 arXiv 标识和完整标题查询引用索引，再核对可读取原文的参考文献，确认以下 **7 篇**包含对 FSG 的引用。引用索引会遗漏或延迟收录，因此这是已核验清单，不是完整引用计数。MAMBO-G 使用的是后续 v4，不能据其最早上传时间推断它在 FSG 发表之前就引用了 FSG。

|直接引用论文|实际研究内容|对本轮的约束或启发|
|---|---|---|
|[CFG-MP，v1](https://arxiv.org/html/2601.21892v1)|将流匹配 CFG 与平滑距离、投影和同伦优化联系起来。|“换成投影或同伦”已有明确先例；本轮必须指出新增目标或约束。|
|[MAMBO-G，v4](https://arxiv.org/html/2508.03442v4)|根据条件差相对于参考速度的幅度，抑制高比例样本的过强 guidance。|仅把标量强度改为状态相关函数不够；需要区分幅度问题和内容判据。|
|[R-Pred，v1](https://arxiv.org/html/2601.22468v1)|使用预训练表示对齐投影器预测参考表示，约束生成中的表示漂移。|中间状态可携带可读的内容估计；本轮没有其投影器，不冒称读到了 DINO 语义。|
|[FDS，v1](https://arxiv.org/html/2604.04646v1)|用流散度关联条件回归残差，比较小扰动候选并修正采样状态。|两分支相等仍可能共同模糊；本轮联合检验一致性与歧义，而不把 FDS 本身重命名。|
|[E2PO，v1](https://arxiv.org/html/2605.15803v1)|在偏好优化中通过条件 embedding 扰动改善探索。|条件空间的扰动是已有工具；其带奖励训练的结论不能直接搬到本轮无训练采样。|
|[GuidedBridge，v1](https://arxiv.org/html/2606.03119v1)|在桥模型中弱化已知先验形成对比，并按频率调节先验引导。|先验是否可读、哪些部分可读是具体对象；桥模型的数据先验与普通高斯 SiT 起点不同。|
|[PathGuide，v1](https://arxiv.org/html/2608.29107v1)|沿当前生成分布，用连续性方程弱形式选择 CFG 强度。|从局部等式到分布路径需要额外条件；没有真实条件场或其估计时，不能借用其路径正确性保证。|

CFG-CTRL 与本问题直接相关：它用误差动力学及控制观点处理条件、无条件预测的趋同。在本次读取的 v2 参考文献中未找到 FSG，因此把它列作相关工作，不计入上述七篇直引清单。其误差衰减思路已经存在，不能将普通比例/微分控制重新算成新想法。[^ctrl]

另两类相关工作帮助确定第二条路线。Saddle-Free Guidance 利用模型的曲率信息避开鞍部；有关 class speciation 的研究用条件熵描述语义承诺的时间区间。它们分别提示“共同模糊”和“何时已经决定类别”，但也说明曲率引导和信息量调度本身并非空白领域。[^saddle][^entropy]

还需避免两处相邻工作重叠。*Commitment Before Realization* 已在 masked diffusion language model 中通过成对未来续生成定义何时可撤除额外 CFG，并指出单步局部统计不能替代该全局判断。因此 #19 的候选贡献只在于连续图像模型的远期残差、重复确认及可恢复迟滞规则，不声称首次提出“类别决定后停止 guidance”。[^commitment] SITCOM 已经为逆问题研究多种扩散一致性及对网络输入的优化；#12 只检验无外部观测条件下的特定自重建代理，不把一致性或输入优化本身说成新概念。[^sitcom]

## 统一符号与可计算对象

小 SiT 采用从噪声到数据的时间约定：`t=0` 是初始高斯噪声，`t=1` 是最终 latent。设 `b=1-t`，`C(z,t)` 为条件速度，`U(z,t)` 为 null 速度，`D=C-U`。普通 CFG 写成 `C+aD`；因此这里的额外强度 `a` 对应常见论文记号 `w=1+a`。混淆这两个记号会导致错误的参数比较。

`m_c=z+bC` 和 `n_c=z-tC` 是当前模型的 signal/noise 预测。`C_H(z)`、`U_H(z)` 是分别沿条件、null 场向前走两次 Euler 的有限区间映射，默认 `H=.25b`；`R_H=C_H-U_H`。它们是有限区间代理，不能称为准确的终点图像。主轨迹用 64 步 Heun，各候选只在指定事件读取这些映射。

状态搜索共享两个方向 `Q=[normalize(D), orthogonalize(xi,D)]`，其中 `xi` 是本次事件固定的高斯探针。共享这个二维搜索空间使目标差异可比较，同时也明确限制了搜索能力。没有把二维子空间求解等同于高维固定点被完整求解。

对于最小二乘候选，在基点冻结归一化、权重和需要保留的目标，实际重新查询 `r(z+epsilon*q_j)`，构造有限差分 `J`。计算

`c = -(J^T J + lambda I)^(-1) J^T r`，`delta=Q c`，

其中 `lambda=.001*mean(diag(J^T J))`，带正数下限，矩阵求解用 FP64。步长上界为 `(4/64)*a*||D||`；少数明确标注的对照或种子约束使用额外倍率。候选状态还要用原来的实际非线性目标重新检查，目标没有下降则保持原状态。这个接受判据只保证**所用代理损失**不增，不能保证 FID 降低。

## 两条路线的理论依据

第一条路线保持“同一起点的续生成应一致”这一目标，改变对解的选择和可信度要求。#1 处理共同漂移，#2 限制相对于冻结 signal 中心的噪声径向变化，#3 同时约束两个时距，#4 检验邻域稳定性，#5 处理空间残差集中，#6 取消已观测到的同场数值往返缺陷，#7 增加中间条件读出，#8 用未来类别证据的梯度拉回替代直接搬运速度差。#8 采用的是局部类别势的一阶信息，并不声称每次更新使全局 FSG 等式残差下降。

第二条路线把“信息已经编码”拆成可以反驳的更具体命题。它可以指条件与 null 的均值和不确定性都接近，也可以指重新加噪后仍可读取、与竞争类别可区分、内容结构一致但外观不必一致，或已经写入的方向不需要再次强化。这些命题彼此并不等价，甚至可能给出相反的实验预测。例如 #9 希望二阶读出一致，#16 则允许两分支不确定性不同，并利用这种差异选择状态。

在理想线性高斯路径 `X_t=tX_1+bZ` 下，可直接推导

`score_c-score_u = (t/b)(C-U)`，

`Cov(X_1|z,c) = (b^2/t) J_z m_c`，`t>0`。

第二式来自后验均值对观测的导数，因此需要实际读取输入 Jacobian。它不允许从两个单独的均值数值反推出任意协方差。本轮 #9 使用真实有限差分方向导数；#16 在固定二维方向上对称化投影 Jacobian，记录负特征值，再采用非负投影和共同尺度的 ridge。最终比较的是受限高斯体积代理，不是无偏互信息估计。流散度与条件残差之间的联系可参见 FDS 的理论部分。[^fds]

同样，在固定未来时间上，`D_future` 与理想类别对数后验的梯度成正比。若通过 null 映射 `Phi_u` 将它用于当前状态，链式法则给出 `J(Phi_u)^T D_future`。这解释了 #8 为什么使用 Jacobian 转置，而不是求逆或直接复制未来向量。实际网络误差、有限区间、二维投影和局部线性化都可能破坏其质量收益。

## 与仓库结果的去重

上一批 50 项包含方向投影、输出方差回缩、patch 增量预算、弱参考空间扰动、历史平滑和 Strong 响应求逆。本轮的残差权重与输出变换必须落在各自明确的**未来等式目标**中，否则仅换名称不算新增。#4 与旧 antithetic 查询共享扰动工具，但本轮搜索的是条件/null续生成一致的状态；#14 对读出目标作商空间归一化，不对最后输出图像作 rescale；#18 搬运的是累计实际写入量，不是 gap 均值。

#6 是明确尝试挽救已知失败机制的候选：同场往返在旧实验中确实能改变轨迹，所以将其作为需扣除的数值缺陷。是否可加、扣除后是否更好需要本轮结果回答。#10 也是明确的组合候选：散度信号已有 FDS 先例，新增检验是它能否排除 FSG 等式代理中的共同模糊状态。

以下逐项列出方程、依据、最近仓库实验、实际差别和局限。每项两种结构参数与五个 CFG 强度全交叉；不会只保留有利的参数点。

### 01　保留条件结局的同起点等式校准

路线 A。**假设：**条件与无条件接近时，应尽量保留原来的条件结局，避免通过共同漂移获得无意义的小残差。

`min_delta ||R(z+delta)||^2 + lambda ||C_H(z+delta)-C_H(z)||^2`

本轮区别：两端均在新状态重新计算，增加条件结局保持项；不同于旧固定目标的单边逆映射。 最近仓库记录：[FSG_ANCHORED_INVERSE_PROBE_20260908_ZH.md](FSG_ANCHORED_INVERSE_PROBE_20260908_ZH.md)。

参数：`conditional_anchor=(0.1, 1.0)`，与五个固定强度交叉，共10组。

局限：条件预测本身可能错误；保持项过强也会阻止有益的模式切换。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[mp](https://arxiv.org/html/2601.21892v1)。

### 02　保留种子噪声长度的等式校准

路线 A。**假设：**写入类别可以主要改变噪声方向；保留当前估计噪声的长度可能减轻径向先验漂移。

`min_delta ||R(z+delta)||^2; delta tangent to n=z-t*C; retract n to its original norm`

本轮区别：先由真实条件/null续生成残差选方向，再在信号中心周围重投影；不是对最终guidance作球面缩放。 最近仓库记录：[FSG_CLOCK_QUALITY_PROTOCOL_20260908_ZH.md](FSG_CLOCK_QUALITY_PROTOCOL_20260908_ZH.md)。

参数：`trust_multiplier=(0.5, 1.0)`，与五个固定强度交叉，共10组。

局限：估计noise不是真实独立高斯；保长度不等于保高斯分布或多样性。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[mp](https://arxiv.org/html/2601.21892v1)。

### 03　同时满足近、远两段续生成等式

路线 A。**假设：**只解一个远期终点可能利用抵消误差；同时约束两个时距能检验写入信息是否在途中也可读。

`min_delta (1-w)||R_H/2(z+delta)||^2+w||R_H(z+delta)||^2`

本轮区别：一个状态更新共同降低两个续生成残差；不是增加同一算子的Picard或Anderson次数。 最近仓库记录：[FSG_ANCHORED_INVERSE_MULTISTEP_20260908_ZH.md](FSG_ANCHORED_INVERSE_MULTISTEP_20260908_ZH.md)。

参数：`long_weight=(0.25, 0.75)`，与五个固定强度交叉，共10组。

局限：两个约束可能冲突；短区间仍可能压过有效的远期语义。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[path](https://arxiv.org/html/2608.29107v1)。

### 04　小扰动邻域内仍成立的等式

路线 A。**假设：**可读取的类别信息应在小噪声扰动下稳定；只在一个精确latent上相等可能过于脆弱。

`min_delta sum_{s in {-1,+1}} ||R(z+delta+s*sigma*(1-t)*xi)||^2/2`

本轮区别：优化扰动下的条件/null续生成一致性；不同于旧IG对扰动处gap取平均。 最近仓库记录：[SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。

参数：`noise_rms=(0.005, 0.02)`，与五个固定强度交叉，共10组。

局限：邻域扰动可能跨语义模式；鲁棒化可能牺牲细节。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[e2po](https://arxiv.org/html/2605.15803v1)。

### 05　优先解决尚未承载条件的空间区域

路线 A。**假设：**全局等式残差可能掩盖少数尚未决定的区域；按基点patch残差加权可把写入预算给这些区域。

`min_delta sum_p softmax(beta*e_p/mean(e))*||R_p(z+delta)||^2`

本轮区别：权重作用于状态搜索的未来等式损失，且每次局部搜索内固定；不是旧patch输出增量裁剪。 最近仓库记录：[SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。

参数：`softmax_temperature=(1.0, 4.0)`，与五个固定强度交叉，共10组。

局限：高残差区域也可能只是噪声或模型误差；没有语义分割监督。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[bridge](https://arxiv.org/html/2606.03119v1)。

### 06　扣除无条件往返的数值漂移后写入

路线 A。**假设：**有用的写入应来自条件差异；纯null离散往返造成的漂移应被独立扣除。

`delta=(L_CU(z)-z)-rho*(L_UU(z)-z), followed by actual-R acceptance`

本轮区别：旧实验只比较同场往返对照，本项显式扣除同场缺陷并检查同起点残差是否下降。 最近仓库记录：[FSG_COMMON_FIELD_CONTROL_PROTOCOL_20260908_ZH.md](FSG_COMMON_FIELD_CONTROL_PROTOCOL_20260908_ZH.md)。

参数：`defect_subtraction=(0.5, 1.0)`，与五个固定强度交叉，共10组。

局限：数值缺陷与条件响应并不严格可加；扣除可能删除偶然有益的数值修正。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)。

### 07　类别强度中间点也应读出同一结局

路线 A。**假设：**若latent已充分承载条件，弱化外部条件也应保持续生成；仅两个端点相等可能隐藏非线性条件响应。

`min_delta ||C_H-U_H||^2+||M_H-U_H||^2, M uses e_u+r(e_c-e_u)`

本轮区别：把中间条件作为额外一致性约束，更新latent；不同于旧CFG条件割线直接替换guidance方向。 最近仓库记录：[SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。

参数：`intermediate_condition=(0.25, 0.75)`，与五个固定强度交叉，共10组。

局限：类别embedding插值未必对应有效概率条件，可能错误限制正常的非线性。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[e2po](https://arxiv.org/html/2605.15803v1)。

### 08　把未来类别证据作为梯度拉回当前状态

路线 A。**假设：**未来的类别证据是需要经Jacobian转置拉回的协向量；直接搬运未来速度差会忽略表示坐标变化。

`delta proportional to Q Q^T J(U_H)^T [C(U_H,s)-U(U_H,s)]`

本轮区别：用有限差分计算未来null流的转置作用在两维搜索子空间中的分量；不同于旧Strong目标的正向响应求逆。 最近仓库记录：[SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。

参数：`horizon_fraction=(0.125, 0.25)`，与五个固定强度交叉，共10组。

局限：未来CFG gap只在理想模型下与类别对数后验梯度成正比；两维投影会遗漏方向。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[path](https://arxiv.org/html/2608.29107v1)。

### 09　条件信息的一阶、二阶读出同时一致

路线 B。**假设：**两个预测均值相同仍可能有不同的不确定性；读取实际输入Jacobian以检验更强的条件承载。

`min_delta ||m_c-m_u||^2+lambda*||J(m_c-m_u)xi||^2`

本轮区别：从真实网络扰动查询估计方向导数，而不从两个均值臆造协方差；旧IG协方差来自训练误差缓存。 最近仓库记录：[IG_HYPOTHESIS_POSTERIOR_RETHINK_20260910_ZH.md](IG_HYPOTHESIS_POSTERIOR_RETHINK_20260910_ZH.md)。

参数：`derivative_weight=(0.1, 1.0)`，与五个固定强度交叉，共10组。

局限：一个方向探针不能识别完整后验；Tweedie协方差解释要求理想线性高斯加噪模型且t>0。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[fds](https://arxiv.org/html/2604.04646v1)、[entropy](https://arxiv.org/html/2602.09651v1)。

### 10　从等式成立但共同模糊的状态中脱离

路线 B。**假设：**条件/null一起落在模糊区域也会接近；一致性需要与条件分支自身的歧义度共同检查。

`choose delta in {0,+r*q,-r*q} minimizing ||R||^2/||R0||^2 + lambda*conditional_divergence`

本轮区别：将FDS的已知散度信号用于FSG候选状态的联合选择；不是把FDS本身算作新方法。 最近仓库记录：[IG_HYPOTHESIS_POSTERIOR_RETHINK_20260910_ZH.md](IG_HYPOTHESIS_POSTERIOR_RETHINK_20260910_ZH.md)。

参数：`ambiguity_weight=(0.1, 1.0)`，与五个固定强度交叉，共10组。

局限：降低局部条件散度不保证FID；随机迹估计有方差，可能偏好收缩。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[fds](https://arxiv.org/html/2604.04646v1)、[saddle](https://arxiv.org/html/2511.21863v1)。

### 11　重新加噪后无条件分支仍能读取预测内容

路线 B。**假设：**稳定的条件内容应进入预测信号，而不只依附于当前特定noise排列；重新加噪提供独立读取测试。

`min_delta ||m_u(s*m_c(z+delta)+(1-s)*xi,s)-m_c(z)||^2`

本轮区别：优化新noise下null读取的固定条件目标；不同于旧IG保持noise长度的旋转平均。 最近仓库记录：[SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。

参数：`future_fraction=(0.25, 0.5)`，与五个固定强度交叉，共10组。

局限：单次再加噪会改变实例细节；强制保持全部latent内容可能过强。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[bridge](https://arxiv.org/html/2606.03119v1)、[repr](https://arxiv.org/html/2601.22468v1)。

### 12　条件预测应是稳定的自重建对象

路线 B。**假设：**即使无需与null完全相同，条件预测也应能沿自身signal/noise分解被再次识别。

`min_delta ||m_c(s*m_c(z+delta)+(1-s)*n_c(z+delta),s)-m_c(z+delta)||^2`

本轮区别：把条件分支自己的重建循环设为latent搜索目标；不同于旧直接外推未来IG gap。 最近仓库记录：[IG_FIXED_POINT_CARRIER_RETHINK_20260909_ZH.md](IG_FIXED_POINT_CARRIER_RETHINK_20260909_ZH.md)。

参数：`future_fraction=(0.25, 0.5)`，与五个固定强度交叉，共10组。

局限：稳定的错误图像也可能满足幂等性；目标本身不能辨别真实图像与模型伪固定点。 相关原文：[repr](https://arxiv.org/html/2601.22468v1)、[mp](https://arxiv.org/html/2601.21892v1)。

### 13　写入目标类别时同时排除最近的竞争类别

路线 B。**假设：**目标/null接近可能仍处于细类别边界；与竞争类别的距离可检验写入是否具有辨别性。

`choose delta minimizing ||U_H-C_H||^2 - lambda*||U_H-V_rival,H||^2`

本轮区别：每类使用冻结embedding余弦最近的另一类，比较未来margin；不同于旧随机类别均值负参考。 最近仓库记录：[SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。

参数：`margin_weight=(0.25, 1.0)`，与五个固定强度交叉，共10组。

局限：embedding最近不一定语义最近；latent距离是margin代理，不是校准类别概率。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[entropy](https://arxiv.org/html/2602.09651v1)。

### 14　只要求内容读出一致，保留通道外观自由度

路线 B。**假设：**像素级等式可能不必要地约束色调和纹理；去通道均值/尺度后的粗空间结构可能足够承载内容。

`min_delta ||P(normalize_channels(C_H))-P(normalize_channels(U_H))||^2`

本轮区别：更改等式的输出空间，再搜索latent；没有修改最终图像方差，也没有使用外部表示模型。 最近仓库记录：[SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。

参数：`pool_size=(4.0, 8.0)`，与五个固定强度交叉，共10组。

局限：粗空间归一化不等于语义表示，可能忽略关键颜色条件或放过伪一致。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[repr](https://arxiv.org/html/2601.22468v1)。

### 15　允许两分支速度不同但走向同一条轨道

路线 B。**假设：**一部分条件/null差异可能仅是沿相同轨道的进度差；不必把这部分解释成未写入的内容。

`min_delta min_tau ||C_H-U_H-tau*U_end||^2+lambda*tau^2*||U_end||^2`

本轮区别：在未来null切向方向消去有惩罚的时间相位，再更新latent；不同于APG对当前clean径向投影。 最近仓库记录：[FSG_CLOCK_QUALITY_PROTOCOL_20260908_ZH.md](FSG_CLOCK_QUALITY_PROTOCOL_20260908_ZH.md)。

参数：`phase_penalty=(0.1, 1.0)`，与五个固定强度交叉，共10组。

局限：一阶时间相位仅局部有效，也可能错误删除真正的类别方向。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[ctrl](https://arxiv.org/html/2603.03281v2)。

### 16　用条件相对不确定性体积识别有信息的状态

路线 B。**假设：**条件信息可以体现在后验体积缩小，未必要求预测均值完全相等；检验信息增益而非均值匹配。

`choose delta minimizing logdet(Sigma_c^Q+ridge*I)-logdet(Sigma_u^Q+ridge*I)`

本轮区别：两维真实Jacobian读出构造受限协方差代理；不是旧固定均值的全局方差最优化。 最近仓库记录：[RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md)。

参数：`covariance_ridge=(0.1, 1.0)`，与五个固定强度交叉，共10组。

局限：仅为两个方向上的高斯代理；非对称/负特征值需投影并记录，不能当作精确互信息。 相关原文：[entropy](https://arxiv.org/html/2602.09651v1)、[fds](https://arxiv.org/html/2604.04646v1)。

### 17　移除前后层重复注入条件产生的非线性交互

路线 B。**假设：**类别信息若已进入前半网络，后半再次注入可能产生过强交互；二因素差分隔离这部分响应。

`I=V_all-V_early-V_late+V_null; v=C+a*(D-rho*I)`

本轮区别：在同一个Full模型中干预前4层与后续层的类别嵌入；不同于旧强弱深度与条件的gap相减。 最近仓库记录：[SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。

参数：`interaction_subtraction=(0.5, 1.0)`，与五个固定强度交叉，共10组。

局限：分层混用类别/null未在训练中出现；交互也可能正是组合语义所需。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[e2po](https://arxiv.org/html/2605.15803v1)。

### 18　只补写当前状态中尚未保留的类别方向

路线 B。**假设：**已写入的类别方向若能被null自然搬运，后续重复强化可能浪费预算并放大伪影。

`carry previous write through J(Phi_u); v=C+a*(D-rho*projection(D,memory))`

本轮区别：记忆保存累计实际guidance写入，并通过null有限差分运输；不是旧gap EMA或双轨影子状态。 最近仓库记录：[IG_FIXED_POINT_CARRIER_RETHINK_20260909_ZH.md](IG_FIXED_POINT_CARRIER_RETHINK_20260909_ZH.md)。

参数：`redundancy_subtraction=(0.5, 1.0)`，与五个固定强度交叉，共10组。

局限：一个累计向量不能代表完整语义记忆，方向运输的线性近似可能失真。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[ctrl](https://arxiv.org/html/2603.03281v2)。

### 19　可恢复的条件写入结束判据

路线 B。**假设：**一旦未来续生成已连续两次接近，就可以停止额外写入；遗忘时应重新开启，而非固定时间永久撤除。

`disable extra CFG after two R_H/(H*||C||) tests below tau; re-enable above 2*tau`

本轮区别：未来一致性、连续两次确认和迟滞共同控制额外CFG；仍保持条件主场，不称永久null接管。 最近仓库记录：[FSG_CONDITION_HANDOFF_RESULTS_20260908_ZH.md](FSG_CONDITION_HANDOFF_RESULTS_20260908_ZH.md)。

参数：`agreement_threshold=(0.05, 0.15)`，与五个固定强度交叉，共10组。

局限：低残差可能是共同模糊；关闭的是额外CFG，不能证明不再需要外部条件。 相关原文：[fsg](https://arxiv.org/html/2510.21512v1)、[ctrl](https://arxiv.org/html/2603.03281v2)、[entropy](https://arxiv.org/html/2602.09651v1)。

### 20　突出类别与空间组织之间的交互信息

路线 B。**假设：**类别信息的一部分依赖长程空间绑定；局部统计保留但空间关系破坏后的条件差可用于隔离它。

`D_bind=D(z)-T^-1 D(Tz); v=C+a*((1-rho)*D+rho*D_bind)`

本轮区别：对输入4x4 latent块作固定per-image置换，对条件/null做双差；不同于旧弱头value置换或局部注意力参考。 最近仓库记录：[SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md](SIT_GUIDANCE_50_IDEAS_CATALOG_20260910_ZH.md)。

参数：`interaction_mix=(0.25, 0.75)`，与五个固定强度交叉，共10组。

局限：置换状态偏离训练分布；隔离出的交互可能是模型对破坏输入的任意响应。 相关原文：[swg](https://arxiv.org/html/2411.10257v3)、[fsg](https://arxiv.org/html/2510.21512v1)。


## 评测与判断

全部 256 组使用一套新产生的 balanced 1K 输入：200 个候选配置和 56 个对照。主要质量比较对象是本轮重新调参后的原生 CFG 以及全部 CFG 对照的最佳结果。IG 曲线、已确认的局部注意力配置另作背景基准，不能将同时调用真实 null 的候选只与纯 IG 比较而宣称大幅改进。

主要输出为 ADM FID，同时保留 sFID、IS、实际 Full/prefix 次数、采样加解码时间、数值失败及候选参数。每个候选的最优值是从十个配置中选择，因此 1K 排名有选择偏差。达到网格边界时只报告固定网格中的胜者。当前自动队列到这次筛选和审计结束，不自动将其称为新 5K 确认或论文成功。

所有完整配置都会进入汇总；数值失败保存为失败记录和空 FID，不从不足 1000 张的子集计算一个“完成结果”。分析重核输入、元数据和覆盖，并对每个 family 的最佳配置额外复核原始与聚合文件哈希、调用成本及缓存特征 FP64 FID/sFID。以上是实验可靠性要求，不能代替方法的理论证明。

## 来源

以下原文均在 2026-09-10 检索或读取。引用关系已核对参考文献；方法说明仅用于其适用范围内。候选公式及质量假设是本轮分析与实现，不归因于所引论文已经提出或验证了同一算法。

[^fsg]: Kaibo Wang 等，*Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations*，NeurIPS 2025，[原文 §3.1–3.2](https://arxiv.org/html/2510.21512v1)。
[^mp]: *Improving Classifier-Free Guidance of Flow Matching via Manifold Projection*，2026，[原文 §3](https://arxiv.org/html/2601.21892v1)。
[^ctrl]: *CFG-Ctrl*，2026，[原文 §3](https://arxiv.org/html/2603.03281v2)。
[^path]: Avishag Nevo、Tamir Hazan，*PathGuide: Dynamic Classifier-Free Guidance via On-Policy Transport Alignment*，2026，[原文 §4 及适用假设](https://arxiv.org/html/2608.29107v1)。
[^fds]: *Training-Free Refinement of Flow Matching with Divergence-based Sampling*，2026，[原文 §3](https://arxiv.org/html/2604.04646v1)。
[^repr]: *Training-Free Representation Guidance for Diffusion Models with a Representation Alignment Projector*，2026，[原文 §3](https://arxiv.org/html/2601.22468v1)。
[^bridge]: *GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance*，2026，[原文 §3](https://arxiv.org/html/2606.03119v1)。
[^e2po]: *Embedding-perturbed Exploration Preference Optimization for Flow Models*，2026，[原文 §4](https://arxiv.org/html/2605.15803v1)。
[^mambo]: *MAMBO-G: Magnitude-Aware Mitigation for Boosted Guidance*，v4，[原文 §3](https://arxiv.org/html/2508.03442v4)。
[^saddle]: *Saddle-Free Guidance: Improved On-Manifold Sampling without Labels or Additional Training*，2025，[原文](https://arxiv.org/html/2511.21863v1)。
[^entropy]: *The Entropic Signature of Class Speciation in Diffusion Models*，2026，[原文 §4](https://arxiv.org/html/2602.09651v1)。
[^swg]: *Sliding Window Guidance*，[原文](https://arxiv.org/html/2411.10257v3)。
[^apg]: *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models*，2024，[APG 原文](https://arxiv.org/html/2410.02416v2)。
[^commitment]: Fan Zhou、Weitian Wang、Tim Van de Cruys，*Commitment Before Realization: When Classifier-Free Guidance Becomes Unnecessary in Masked Diffusion Language Models*，2026，[原文 §3–4](https://arxiv.org/html/2608.08082v1)。
[^sitcom]: Ismail Alkhouri 等，*SITCOM: Step-wise Triple-Consistent Diffusion Sampling for Inverse Problems*，2024，[原文](https://arxiv.org/abs/2410.04479)。
