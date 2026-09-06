# 固定预训练特征的实际轨迹密度比

2026-09-07。3% FID目标仍未达到。本方向在原生实际轨迹数据上保留同一个密度比反馈公式；小数据三次拟合全部未通过独立验证，没有对应FID。当前进行一次64K/8K扩容，不搜索层、时间窗、温度或外推系数。

## 论文依据与具体近似

再次核对 [Discriminator Guidance 原文 §5.1](https://proceedings.mlr.press/v202/kim23i/kim23i.pdf)：论文的大部分实验在冻结的预训练噪声分类器特征上训练较浅判别器；其DiT与LSGM设置还有单独说明，并非所有实验完全相同。因此，从零训练小型原始latent网络不是迁移该论文的唯一实现。这里利用现成RAEv2模型的固定base_model_depth=8作为预训练前缀，没有根据验证损失或FID挑层；它是迁移假设，不声称RAEv2前缀等价于该论文的噪声分类器。

每个patch token的1440维前缀输出h先作自身RMS归一化：

    r_ij = h_ij / sqrt(mean_j(h_ij²) + 1e−8),
    phi(z,t,c) = concat(mean_i r_ij, mean_i r_ij²),  dim(phi)=2880.

一、二阶统计定义了预训练token坐标中的低阶、对角二次指数族。条件c和t通过原模型前缀进入特征。取单个线性头f=w·phi_tilde+b，logit d=(1−t)f。它只有2881个参数。即使特征空间密度比能准确拟合，提升到z的梯度也一般只是原密度比梯度的近似；池化丢失空间关系，所有时间/类别共享一个头也限制表达能力。不假设这些特征充分，也没有FID保证。

FP32提取、TF32关闭。所有小bank特征worker均独立比较standalone前缀与原完整模型forward hook，逐位一致；原官方采样仍保持原BF16算术。

## 共享噪声凸目标与固定先验

令a=1−t，p为a X_real+t epsilon，q为从同一epsilon出发的原官方实际query状态。使用同一scaled logistic目标：

    ell(f_p,f_q,a) = (f_q−f_p)/(4a)
        + [log cosh(a f_p/2)+log cosh(a f_q/2)]/(2a²).

在无限表达能力和正确分布下，逐t、c的未正则化最优logit仍为log(p/q)。共同epsilon使两状态在a趋零时差为O(a)，消除线性项的总体边界发散；FP32误差和有限网格并不因此消失。实际轨迹的边界推导、局部KL耗散项及仍存在的advection/冻结q失配见 [实际分布推导](RAEV2_ACTUAL_RATIO_20260907_ZH.md)。

用训练positive/negative各半权混合分布估计一个2880维均值mu和**一个标量**RMS sigma，然后附加常数1。由构造有 E||phi_aug||²=d=2881。固定广义Bayes先验 w_aug~N(0,I/d)使先验平均 E f²=1。对N个独立生成状态的平均风险，使用lambda=d/N：

    J(w) = mean_n mean_k ell(w·p_nk, w·q_n, a_n) + d/(2N)||w||².

这是有明确尺度解释的建模选择，不是唯一正确的理论正则强度。bias也在同一先验中。损失关于w凸，仅从零进行一次L-BFGS-B求解，maxiter500、gtol1e−8、ftol1e−13；不根据验证/FID试lambda。独立autograd检查验证了目标、梯度和先验只计入一次。

## 小数据全部结果

原bank为5000个训练native状态/1000个独立验证状态，真实样本每类5/1张。

| 方法 | 参数数 | 训练风险 | 验证风险 | 类SE | 验证风险+2SE | 采样准入 |
|---|---:|---:|---:|---:|---:|---|
| 实际轨迹小Transformer | 762753 | −1.086087（最后128更新均值） | −.029079 | .144497 | +.259915 | 失败 |
| 冻结8层特征，单配对凸头 | 2881 | −4.186330 | +1.321792 | .685660 | +2.693113 | 失败 |
| 同一特征，五real条件平均凸头 | 2881 | −3.366746 | +.937831 | .624391 | +2.186613 | 失败 |

这些训练风险来自不同目标/求值集合，不能直接将三者差解释成机制贡献。原单配对头在全部五real的条件平均训练风险为−2.944208，新的条件平均头为−3.366746；后者验证仍高于零。两个凸解的最大梯度分别2.75e−6和1.27e−6，已收敛，不能把结果归于优化器尚未完成。

条件平均在固定q/epsilon/time/class下对原5个真实选择**全部枚举**；依据条件期望的方差分解，相对于独立随机抽一个real可减少该条件随机性。旧单配对是固定分层配对，观察到的风险差不能直接等同于理论方差减少量。q特征和旧选中p特征在5000个id上全部逐位一致。五种配对不是25000条独立轨迹，lambda仍为2881/5000=.5762。

对全部99个时刻等权的描述性审计仍未给出通过证据；保留原始准入门槛，未择窗口。重新检查真实bank存储顺序与PackedImageNet行号/类别，未发现错配。细节见 [时间审计](../experiments/results/raev2_guidance_20260907/actual_ratio_small_time_audit.json)。固定小样本结果：[actual fit](../experiments/results/raev2_guidance_20260907/actual_ratio_fit.json)、[prefix单配对](../experiments/results/raev2_guidance_20260907/prefix_ratio_fit.json)、[prefix条件平均](../experiments/results/raev2_guidance_20260907/prefix_ratio_rb_fit.json)、[条件特征身份](../experiments/results/raev2_guidance_20260907/prefix_ratio_rb_features.json)。

## 一次64K/8K固定扩容

训练和验证风险有明显差距。N=5000与2881个参数的比例提供扩容动机，不能证明样本数是唯一原因。固定扩到64000训练/8000验证，沿用相同前缀、统计、归一化、先验公式和guidance强度。这是一次工程预算，不声称64K由理论唯一导出，不安排16K/32K/64K比较或按FID挑数据规模。

- Native train/validation seed202609101/202609102，query排列seed202609103/202609104。每个id仍只有一个预先选定的query，B8、原官方100步shift8。FP32保存state和epsilon。无decoder和FID。
- 真实图像seed202609105，每类64train/8validation，无放回抽取；两split与当前旧6000个real行无重叠，72000行唯一。不声称整个历史仓库从未使用过这些公开训练图像。
- 同一PackedImageNet原始图像、ADM确定性center_crop256、无flip、FP32编码、原K7归一化一次、FP16存储。每个real worker先用原bank的首个B8检查重新编码，保存latent逐位一致；父进程检查各worker互不重叠的写入片段哈希和完整文件哈希。真实bank已完成，见 [身份](../experiments/results/raev2_guidance_20260907/real_ratio_bank64k.json)。
- 训练每个native id使用k=0..4的五个real：real_round=(id//1000+k) mod64。每类每个real恰好出现五次；顺序已独立随机化，不依赖noise。这里只平均五个平衡选择，**不是全64枚举**。
- 验证每条native noise使用该类全部八个独立held-out real，先在real选择上平均，再按1000个类求SE。独立native数量为8000，不能按64000配对求IID SE。SE是固定类平衡设计下的描述性统计，不给形式化泛化保证。
- 正则lambda=2881/64000=.045015625；数值随N按旧规则变化，不是搜索后的lambda。仍单次L-BFGS-B、零初始化、同一收敛条件。保留未正则验证风险均值+2类SE<0的门槛。
- 只有通过门槛后，才审计实际输入梯度、完整有限轨迹和original official8逐像素一致性，再进行强度1、全100时刻、seed202609071的固定1K。拟合数据、特征和分类风险不使用FID。通过数值、像素与完整1K检查后，按首次FID前追加冻结的决定，用同一head进行一次独立seed202609072的5K，是否运行不由1K分数决定。若满足质量目标，再核验实测成本对照；不更换原官方基线。

实现：[固定计划](../experiments/raev2_prefix_ratio64k_plan.py)、[数据到拟合的单次执行器](../experiments/prepare_raev2_prefix_ratio64k.py)、[特征提取](../experiments/extract_raev2_prefix_ratio64k.py)、[凸头拟合](../experiments/fit_raev2_prefix_ratio64k.py)。执行器只等待现有native进程的PID+starttime，不重启该数据生成；先全量核验输入文件，再启动特征提取。失败保留输出并停止，不自动追加另一轮拟合。

数据准备期间real编码曾与native生成共享四GPU，因此这两部分worker秒含并发等待，不能将其和标成独占GPU计算时间。主模型逐样本调用数、特征前缀调用数、训练时间、正式采样/解码时间分别保存。数据扩容本身不是目标完成。

## 当前实现校验与执行位置

四项CPU解析测试通过：原单配对梯度、原条件平均梯度、新缓存线性项的等价梯度，以及64K配对的类内精确计数/先验有效N。新GPU输入梯度审计已写好，**尚未运行**，也未把小数据失败head绕过准入用于采样。

RAEv2 RMSNorm.forward内部强制`.float()`；普通`model.double()`不会构成完整FP64参考。预定审计保留同一权重和RoPE常量，在独立副本中使用FP64 RMS计算和FP64零attention mask，对原FP32部署梯度作比较，再用FP64中心差分独立验证；阈值固定为相对误差.002。正式模型不作这些参考计算替换。另检查外层no_grad/BF16 autocast环境中的梯度、TF32开关恢复、纯噪声时刻有限和零头精确零修正。见 [待执行梯度审计](../experiments/audit_raev2_prefix_ratio64k_gradient.py)。

单次后续进程PID2145485等待现有native父进程PID2072122的starttime身份209554353，未重启生成。实时状态在 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/prefix_ratio64k_features/execution.json`；Git中的 [启动快照](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_pipeline_snapshot.json) 明确标为未完成，不代表之后的实时进度。数据输入计划见 [64K native计划](../experiments/results/raev2_guidance_20260907/actual_ratio_bank64k_plan.json) 与 [固定特征/拟合计划](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_plan.json)。

采样集成补丁位于 `experiments/locks/raev2_prefix_ratio64k_20260907`，`git apply --check`通过，**尚未应用**；此时native/feature进程读取的原采样源码仍冻结。只有两项bank、特征和固定拟合完成并通过准入后才应用补丁，随后由 [固定screen执行器](../experiments/run_raev2_prefix_ratio64k_screen.py) 顺序运行梯度审计、original8/candidate8、paired1K。主模型与额外8层前缀forward/backward分别计数，采样实测秒包含全部guidance工作。执行器不会自行宣布goal完成。


## 在新head与首次FID前冻结的独立5K

当前用户目标允许1K或5K，而仓库已观察到两个N下方法排序变化。因此在64K拟合尚未产生head、且新FID尚未出现时，固定同一个模型完成paired1K后，再作一次独立seed202609072的5K；这次5K不以1K质量正负为条件。前置分类准入、实际输入梯度、original8逐像素一致性和完整有限1K仍须通过。两种N完整报告；不存在按FID改head、先验、强度、层或时间窗的分支。

5K复用已经完成的official100控制：FID6.9497684777115865，初始noise/label SHA `b59864ce96fcfb63735061f00ecb903b07ad894ffb504a73918d7b704db86cc8`。原官方参考、类别、B8与采样网格不变。完整条件和决定时间见 [首次head/FID前计划](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_quality_plan.json)。这是同一固定候选的两项评测，不是第二次训练。

[后续执行器](../experiments/continue_raev2_prefix_ratio64k_quality.py)已经启动，等待现有特征/拟合父进程的PID+starttime消失；它不启动第二个拟合或重建bank。拟合未过门槛则结束且不应用采样补丁；通过后才核验并应用已准备的唯一补丁，顺序执行梯度→original8/candidate8→paired1K→独立5K。任何阶段失败保留原目录并停止，不自动重跑。若出现3%质量结果，独立FID复算与适当成本对照仍由后续审查完成，队列不会自行改goal状态。

另有一项新增CPU数值fixture检查通过：实际DDT模块的小尺寸、未训练架构的FP32输入梯度与独立FP64副本相符，FP64中心差分也通过。它专门检验参考实现的RMS精度与mask，不是已训练RAEv2梯度或质量结果。连同之前四项解析测试共五项通过。环境版本和GPU型号已实际读取并记录于 [复现环境](../experiments/results/raev2_guidance_20260907/prefix_ratio64k_environment.json)。


## 待执行的质量复核

[质量审计器](../experiments/audit_raev2_prefix_ratio64k_quality.py)已准备好，当前尚无候选图像，未运行其正式入口。它仅使用最终全量图像和固定官方Inception特征：核验global ids、每批noise/labels、模型/config/decoder/stats、head与源码快照；比较合并图像与四分片的全部像素。1K用N×N Gram特征值，5K用对称参考协方差根形式，均使用FP64和Bessel协方差，独立于官方scipy.sqrtm路径。两条数学路径已在具有已知对角Gaussian闭式FID的确定性点集上通过检查，包含低秩情形；这不是实际候选FID。

已完成基线的独立复算不重复运行，而是校验先前审计及其样本/特征哈希。输出同时保留official与历史interval的原FID、两种相对改善、分开的数据准备/拟合/推理成本；不把曾经并发的native/real worker秒相加冒充独占GPU计算。满足质量门槛后，成本对照仍须完成。
