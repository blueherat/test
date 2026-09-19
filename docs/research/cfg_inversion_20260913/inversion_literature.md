# 流匹配反演：操作、离散误差与 CFG 条件

流匹配有明确的“沿同一个生成系统返回”操作：保持网络、条件和 guidance 规则一致，反向求解其 ODE。它可以作为重建和反演研究的对象，但反向数值求解不等于精确撤销离散采样，也不保证图像质量改善。

本文统一用 **t=0 为噪声、t=1 为图像**。论文、FLUX 代码常采用相反端点或相反索引，下面公式先统一坐标，再讨论具体实现。论文日期采用 arXiv 首次提交日，会议年份另列。

## 1. 可以直接使用的两种反演写法

设条件固定为 c，普通 conditional 场为 v_c，无条件场为 v_u，gap 为 g=v_c−v_u；总 CFG 权重为 w：

\[
v_w(x,t;c)=v_u(x,t)+w(t)g(x,t;c)
          =v_c(x,t;c)+(w(t)-1)g(x,t;c).
\]

这里 w=1 是普通 conditional。生成满足 dx/dt=v_w(x,t;c)。从图像 latent x_1 反演，有两种等价的连续写法：

\[
\boxed{\text{用同一个 }v_w\text{，将积分时间从 }1\text{降到 }0.}
\]

或设置新的递增时钟 s=1−t：

\[
\boxed{y_0=x_1,\qquad \frac{dy_s}{ds}=-v_w(y_s,1-s;c),\quad s:0\to1.}
\]

因此，“负速度”需要同时说清楚模型收到的时间。**不能既反转时间网格，又额外把速度取负**，否则符号抵消。每一步应在当前逆向状态重新评估网络；guidance schedule 按原物理时间 t 查询，而不是重新从第一个采样步开始播放。RF-Inversion 的 Proposition 3.1 明确写出了 −v_{1−t} 与原 ODE 的对应关系。[^5]

设学习到的 v_w 在所考虑区间产生唯一解，且正反轨迹均存在。精确解的流映射满足

\[
\Phi^w_{a\leftarrow b}\circ\Phi^w_{b\leftarrow a}=I.
\]

这是对**同一个学习场**的结论，不要求这个网络已经学到真实数据分布。模型不够好可以导致图片不自然、反演噪声不典型或数值求解困难，但“偏离真实速度”本身不自动破坏其确定性 ODE 的可逆性。若包含历史控制器，状态应扩展为 (x,history)；仅保存 x 后清空历史，不属于同一个系统的逆。

## 2. 反走一步，与精确撤销一步不同

下面是独立推导。若生成用显式 Euler：

\[
y=x+h v_w(x,t),\qquad h>0,
\]

从 y 反向 Euler 得到

\[
\widetilde x=y-hv_w(y,t+h).
\]

精确撤销原 Euler 步则需要解未知 x：

\[
\boxed{x=y-hv_w(x,t).}
\]

这不是在 y 上再预测一次就完成。一个直接的固定点迭代是

\[
x^{(k+1)}=y-hv_w(x^{(k)},t).
\]

在适当自映射区域且 hL_w<1 时，它是压缩映射；有限迭代不是无条件精确逆，失败时也不能把数值残差解释为生成质量。ReNoise 正是沿这类隐式逆方程反复更新未知状态预测，而不是每次向图片注入独立高斯噪声。[^4]

两个一维反例能分清对象：

- 若 v(x)=λx，精确 ODE 正反组合为 e^{−λh}e^{λh}x=x；Euler 正反组合为 (1−λ²h²)x，通常不回原点。
- 若 λ=−1/h，原 Euler 步把所有 x 变成0，离散映射根本没有逆；但对应连续 ODE 仍为可逆的指数缩放。

记录当初的速度并做 y−h·v_cached 可以撤销那次算术更新，但使用了额外轨迹记录；它不等同于从当前图片独立推断逆状态。

## 3. 七篇最相关文献的操作对照

|文献与首发日期|真正做的操作|条件、guidance、额外状态|是精确离散逆吗？|
|---|---|---|---|
|DDIM，2020-10-06；ICLR 2021[^1]|η=0 的确定性采样；将对应 ODE 数值路径反走以编码，再生成重建。§5.4 报告步数增多时重建误差下降。|原论文不是高 CFG inversion 专门方法。迁移到条件模型时须明确源条件和 CFG schedule。|普通反向 Euler/显式 DDIM inversion 只是近似。|
|Null-text inversion，2022-11-17；CVPR 2023[^2]|先以 w=1 做 DDIM pivot inversion，再以 w=7.5 逐时间优化空文本 embedding，使生成靠近 pivot。|条件文本及网络冻结；保存源图特定的逐时间 null embeddings。生成场被优化过。|是优化得到的重建，不是把原 w=7.5 映射代数求逆。|
|EDICT，2022-11-22；CVPR 2023[^3]|引入两个耦合 latent，交替作关于其中一个 latent 的仿射变换；逆向按相反顺序解出两者。|必须保留成对状态、相同预测函数/条件/系数；编辑改变目标条件。|其新构造的双状态离散过程代数可逆；不是证明原单状态 DDIM 已精确可逆。|
|ReNoise，2024-03-21；ECCV 2024[^4]|每个目标噪声时间 t 内，反复在当前猜测 z_t^(k) 上评估网络，修正隐式逆；再平均后几次预测。|固定目标 t 和对应源条件；随机 sampler 的外部噪声必须记录并复用。完整编辑版本还有正则化及 noise correction。|固定点收敛时可求对应逆方程；有限迭代、编辑正则和随机修正没有统一的无条件精确保证。|
|RF-Inversion，2024-10-14；ICLR 2025[^5]|区分纯 ODE 逆与引入 LQR 控制的反演/重建；控制项在场与指定端点方向间插值。|受控版本引入指定噪声端点，返回时还可显式使用原图作为吸引目标。|无控制、同场精确积分具有连续逆；受控实际方法换了场，不能将其忠实度归因于纯逆积分。|
|RF-Solver，2024-11-07；ICML 2025[^6]|用沿轨迹的二阶 Taylor 项；实际二阶实现是显式 midpoint，每步重评估中点；反演反转网格。|RF-Edit 另行缓存并注入 attention V。FLUX-dev 的 guidance 是蒸馏模型输入，不直接等同双分支 CFG。|减少截断误差的数值方法，非 EDICT 式代数可逆。|
|FireFlow，2024-12-10；ICML 2025[^7]|首步做 midpoint，随后用上一步缓存的中点速度构造当前中点，再运行当前中点网络。|官方重建例同 source/target prompt、guidance=1、inject=0；编辑则改变条件、guidance并可注入V。|近似求解；缓存复用不是精确撤销此前离散步。|

## 4. 各类方法最值得借用的公式

### DDIM / ReNoise：预测的位置才是关键

用扩散文献的原生索引，t 较大表示较高噪声。η=0 的 DDIM 一步可以写成

\[
z_{t-1}=a_tz_t+b_t\epsilon_w(z_t,t;c),
\]

\[
a_t=\sqrt{\bar\alpha_{t-1}/\bar\alpha_t},\qquad
b_t=\sqrt{1-\bar\alpha_{t-1}}-a_t\sqrt{1-\bar\alpha_t}.
\]

其逆方程为 z_t=[z_{t−1}−b_t ε_w(z_t,t;c)]/a_t。常见 DDIM inversion 将未知 z_t 处的预测替换为已知较干净状态处的预测；不同实现的模型时间索引也可能不同，不能只写“取负噪声”。ReNoise Eq.(1–4) 明确保留目标 t，对当前 z_t^(k) 重新预测：

\[
z_t^{(k+1)}=
\frac{z_{t-1}-\psi_t\epsilon_\theta(z_t^{(k)},t,c)-\rho_t\xi_t}{\phi_t}.
\]

若 ρ_t≠0，ξ_t 是该步记录的外部噪声，两方向复用；它并非在每次固定点迭代独立抽一个新噪声。完整 ReNoise 将重建与可编辑性一起考虑，不能把额外正则和 noise correction 的效果全部归为 ODE 求解精度。[^1][^4]

### Null-text：低 CFG 反演、高 CFG 返回需要补偿

其 Eq.(3)、Algorithm 1 的核心是

\[
\min_{\varnothing_t}\;
\|z^*_{t-1}-D_{t,w=7.5}(\bar z_t;c,\varnothing_t)\|^2.
\]

z* 来自 w=1 的 pivot。每步优化后，用实际生成结果更新下一步的 bar-z，而不是假装它已经等于 pivot。它是“不同 guidance 之间用源图特定参数补偿”的现成例子，不能被概括为低 CFG 反演天然就是高 CFG 采样的逆。[^2]

### EDICT：通过扩展状态让离散更新可逆

略去另加的可逆 mixing 层，其 Eq.(10–11) 为

\[
x'=a x+b\epsilon(y,t),\qquad y'=a y+b\epsilon(x',t),
\]

\[
y=(y'-b\epsilon(x',t))/a,\qquad
x=(x'-b\epsilon(y,t))/a.
\]

重新查询网络的位置在反向运算中可恢复，因此无需把 ε(y,t) 近似成 ε(x,t)。同一函数、相同条件和非零 a 是这里的前提。作者也明确指出：代数可逆不自动带来真实感，mixing 的逆会放大浮点误差；VAE 编码本身的像素损失也不属于这个保证。[^3]

### RF-Solver / FireFlow：改善积分精度，而不宣称精确撤销

用有符号步长 h，RF-Solver 二阶版本为

\[
k_1=v(x,t),\quad x_m=x+\frac h2k_1,
\quad k_2=v(x_m,t+h/2),\quad x'=x+h k_2.
\]

这是沿轨迹总导数的有限差分，包含状态变化；不是仅对 t 做偏导。FireFlow 在首步以后以此前保存的中点速度代替 k_1，当前 k_2 仍重新评估并存储。两者都可用有符号网格反向求解；不能仅将已保存的所有速度倒序播放来声称完成独立反演。[^6][^7]

官方 FireFlow `src/edit.py` 把 inversion 的 `guidance` 固定为1，生成使用命令行 guidance；`src/flux/sampling.py` 在 `inverse=True` 时反转网格，并用 `t_prev-t_curr` 更新。网络只收到一条 `guidance_vec`，没有在此处计算 conditional/unconditional 两支的差。重建示例的 `guidance=1`、相同 prompt 与关闭注入很关键；编辑示例不是相同场的往返。[^8]

RF-Solver 官方 `FLUX_Image_Edit/src/edit.py` 也在反演调用中固定 `guidance=1`，随后以目标 prompt 和命令行 guidance 生成。其 `sampling.py` 的默认模型时间由1降到0；反演时反转为0到1。因此这份 FLUX 代码的数值时间与本文统一的 SiT 约定相反，移植时不能仅复制时间数字。[^9]

## 5. 高 CFG 为什么会更难重建，却仍可能有连续逆？

Null-text 和 EDICT 都报告 CFG 放大近似反演中的误差。这里应分开两种来源：同一大 w 的场在数值上更难求解；以及两方向使用不同 w 或文本/空文本条件，导致它们根本不是同一场。[^2][^3]

下面是对 CFG 的独立分析，而非引用某篇论文的保证：

\[
Jv_w=Jv_u+w(Jv_c-Jv_u).
\]

较大 w 可能增加局部变化率、条件数或误差传播；它不必在所有方向都增加，也不自动破坏唯一解。一般 Lipschitz 条件可给误差随路径长度按 exp(∫L dt) 放大的上界；要声称逆向收缩，需要更强的单边收缩/耗散条件。

**FireFlow 的一个不能直接沿用的理论结论。** ICML 正式版本 Proposition 3.1 / Appendix B.1 仅从 Lipschitz 条件推得逆向误差上界 e^(−LT)。证明把不等式从 T 积分到0时未处理方向变化。反例 v(x)=−Lx 的正向流收缩 e^(−LT)，其逆流恰好放大 e^(LT)，因此该假设不足以推出论文给出的普遍收缩式。应保留其算法及实验贡献，避免借该命题声称“反走一定更稳定”。这是独立核查结论。[^7]

## 6. 高 CFG 前进，再以 conditional 反走是什么？

以下是独立局部展开。先用 v_w 前进一个小步，再用普通 conditional 场反向一步：

\[
y=x+h[v_c(x,t)+(w-1)g(x,t)],
\qquad x_{\mathrm{back}}=y-hv_c(y,t+h).
\]

因此

\[
\boxed{x_{\mathrm{back}}-x=h(w-1)g(x,t)+O(h^2).}
\]

即使两段都精确积分，组合 Φ^c_{t←t+h}∘Φ^w_{t+h←t} 通常也不是恒等。第一阶效应就是留下 CFG 的额外推进，之后才有状态变化和场不交换的二阶信息。不能将其残差直接命名为反演失败、质量缺陷或自动纠错。

若希望把这种操作与 FSG 一类重新评估场的方法连接，值得研究的是**扣除 h(w−1)g 后的非平凡残差**，以及它在不同 h、solver 和等计算预算下是否稳定。至少同时比较同 w 往返、w→1 往返和直接施加 h(w−1)g；否则容易把普通 guidance 重写成一个额外耗时的回路。

## 7. 与 fresh 加噪的比较应按任务确定

对当前原图 x_1，FM 的 fresh noising 通常是 z_τ=τx_1+(1−τ)ξ，ξ 为新高斯样本；纯反演是 z_τ=Φ^w_{τ←1}(x_1)。前者指定随机条件分布，后者指定模型流中的确定性坐标，二者不是同一操作。

若目标是沿同场返回原 latent，反演给出了明确的连续恒等对象；fresh noising 没有逐图恒等承诺。若目标是编辑、去除输入缺陷或引入多样性，精确逆会保留输入缺陷，不构成一般优势。RF-Inversion 正因为区分“忠实返回”和“转向更典型样本”，才另外加入控制项；EDICT 也明确区分可逆性与结果实用性。[^3][^5]

对外报告应分别说明：latent round-trip 残差、VAE 像素误差、源内容保持、编辑完成度和额外 NFE。不能用一个更低的循环误差推出更好的生成分布，更不能据此比较强弱模型的生成能力。

## 原始来源

[^1]: Jiaming Song, Chenlin Meng, Stefano Ermon. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502). 首发2020-10-06，ICLR 2021。操作定位：[正文 §4.3、§5.4](https://arxiv.org/html/2010.02502v3)。
[^2]: Ron Mokady et al. [Null-text Inversion for Editing Real Images using Guided Diffusion Models](https://arxiv.org/abs/2211.09794). 首发2022-11-17，CVPR 2023。操作定位：[§3.2、Eq.(3)、Algorithm 1、Appendix B](https://arxiv.org/html/2211.09794v1)。
[^3]: Bram Wallace, Akash Gokul, Nikhil Naik. [EDICT: Exact Diffusion Inversion via Coupled Transformations](https://arxiv.org/abs/2211.12446). 首发2022-11-22，CVPR 2023。操作定位：[§4.1–4.3、Eq.(10–15)](https://arxiv.org/html/2211.12446v1)。
[^4]: Daniel Garibi et al. [ReNoise: Real Image Inversion Through Iterative Noising](https://arxiv.org/abs/2403.14602). 首发2024-03-21，ECCV 2024。操作定位：[§3.1、Eq.(1–4)、Algorithm 1、Appendix D](https://arxiv.org/html/2403.14602v1)；[官方项目](https://garibida.github.io/ReNoise-Inversion/)。
[^5]: Litu Rout et al. [Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations](https://arxiv.org/abs/2410.10792). 首发2024-10-14，官方仓库标注 ICLR 2025。操作定位：[§3.2 Proposition 3.1、§3.3 Eq.(8)、§3.5](https://arxiv.org/html/2410.10792v1)；[官方代码](https://github.com/LituRout/RF-Inversion)。
[^6]: Jiangshan Wang et al. [Taming Rectified Flow for Inversion and Editing](https://arxiv.org/abs/2411.04746). 首发2024-11-07；[ICML 2025正式出版](https://proceedings.mlr.press/v267/wang25ce.html)。操作定位：[§3.2 Eq.(10–12)、Algorithm 1、§4.1](https://arxiv.org/html/2411.04746v1)；[官方代码](https://github.com/wangjiangshan0725/RF-Solver-Edit)。
[^7]: Yingying Deng et al. [FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing](https://arxiv.org/abs/2412.07517). 首发2024-12-10；[ICML 2025正式出版](https://proceedings.mlr.press/v267/deng25c.html)。操作定位：[§4 Eq.(10–12)、Appendix A Algorithms 1–2](https://arxiv.org/html/2412.07517v1)。理论核查使用[正式PDF，Proposition 3.1和Appendix B.1](https://raw.githubusercontent.com/mlresearch/v267/main/assets/deng25c/deng25c.pdf)。
[^8]: FireFlow 作者仓库：[重建与编辑示例](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing#3-demo-scripts-inversion-and-reconstruction)、[`src/edit.py`](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/blob/main/src/edit.py)、[`src/flux/sampling.py`](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/blob/main/src/flux/sampling.py)。读取日期2026-09-13；main链接可变，以上具体实现位置为当日读取版本。
[^9]: RF-Solver 作者仓库：[`FLUX_Image_Edit/src/edit.py`](https://github.com/wangjiangshan0725/RF-Solver-Edit/blob/main/FLUX_Image_Edit/src/edit.py) 第157、164行；[`FLUX_Image_Edit/src/flux/sampling.py`](https://github.com/wangjiangshan0725/RF-Solver-Edit/blob/main/FLUX_Image_Edit/src/flux/sampling.py) 第66、94–97、118–134行。读取日期2026-09-13，main链接可变。
