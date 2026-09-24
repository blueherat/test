# 主图英文图注

**Overview of endpoint-supervised self-guidance learning.**
A frozen strong predictor $S$, a weak predictor $W_\theta$, and signed time-dependent coefficients $a_\alpha(t)$ define the guided prediction $S+a_\alpha(t)(S-W_\theta)$ throughout sampling. The terminal state is mapped to an RGB image $x_f$, encoded by a frozen Inception network $\phi$, and scored by a class-conditional real/fake classifier $D_\omega$.
**(a) Guidance update.** With the classifier fixed, we update the weak predictor and guidance coefficients using the non-saturating adversarial objective and a soft gap-energy anchor on independent real-data probes. Gradients propagate through the image decoder, feature encoder, classifier, and all steps of the discrete sampling trajectory; freezing model weights preserves their input derivatives.
**(b) Discriminator update.** With the sampling parameters fixed and generated samples detached, we train the classifier on real and generated endpoint features with logistic loss and real-feature R1 regularization. The bottom illustrations depict endpoint matching and a changing decision boundary in a fixed feature space, respectively; they are conceptual rather than measured distributions. Training alternates a discriminator update followed by a guidance update using the updated classifier.

## Symbols and objective definitions

The classifier outputs a **logit**, not a probability:

\[
d_r = D_\omega(\phi(x_r),c),\qquad
d_f = D_\omega(\phi(x_f),c),\qquad
\operatorname{softplus}(s)=\log(1+e^s).
\]

The two objectives are

\[
\mathcal L_D = \mathbb E[\operatorname{softplus}(-d_r)]
+\mathbb E[\operatorname{softplus}(d_f)]
+\frac{\gamma}{2}R_1,
\qquad
R_1=\mathbb E_{x_r,c}\!\left[
\left\|\nabla_h D_\omega(h,c)\big|_{h=\phi(x_r)}\right\|_2^2\right],
\]

\[
\mathcal L_{\rm guidance}
=\mathbb E[\operatorname{softplus}(-d_f)]+\lambda R_{\rm gap}.
\]

For the joint SiT implementation, $W_0$ is the frozen initial weak predictor.
On a fresh probe minibatch $\mathcal B$ of real-posterior interpolants, define

\[
E_\theta(\mathcal B)=\operatorname{mean}_{\mathcal B,\,\mathrm{coordinates}}
  (S-W_\theta)^2,
\qquad
R_{\rm gap}
=\mathbb E_{\mathcal B}\left[
\log^2\!\frac{E_\theta(\mathcal B)+\epsilon}
               {E_0(\mathcal B)+\epsilon}\right].
\]

This is an expectation of a nonlinear minibatch ratio, not a ratio of population expectations. The probe states are independent of the learned sampling dynamics. Current defaults are $\lambda=0.1$, $\epsilon=10^{-8}$, and $\gamma=1$.
The anchor controls a sampled residual scale; the figure does not assert parameter identifiability or guaranteed distributional convergence.

## Schedule-only version

For the second figure, replace the guidance-update sentence with:

> With both strong and weak predictors fixed, we update only the signed guidance coefficients through the entire sampling trajectory. No weak-head update or gap-energy anchor is applied.

In the SiT joint figure, $S$ and $W_\theta$ predict velocity and a frozen VAE maps the terminal latent to RGB. In the JiT schedule-only figure they predict clean data, the guided prediction is converted to velocity inside the sampler, and a fixed rescaling/clamping map produces RGB. Thus the expression inside the sampler is labeled as a prediction mixture rather than a universal velocity equation. $a_\alpha(t)$ is implemented as a learned signed scalar for each solver interval, not a state-conditioned network.

## 中文说明

主图左侧表示学习弱头与时间系数，右侧表示学习真假分类器。强生成模型、Inception 和图像映射始终固定。冻结的是参数；左侧对输入和完整采样轨迹的梯度仍然保留。右侧只改变分类边界，不改变冻结表征中的样本坐标。主图中的探针锚点对应当前 SiT 联合训练；第二页对应当前 JiT 仅学习系数的配置。

Small network, feature-map and schedule icons are schematic: their displayed node counts, layer counts and curve shapes do not specify the precise architecture or report learned parameter values. All labels, formulas and Greek symbols use Comic Sans MS to match the reference figure's lettering style.
