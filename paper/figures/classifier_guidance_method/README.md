# 我们的方法图：分类器反馈学习自引导

延续 AdvFD Figure 3 的双栏布局、蓝／金训练状态、浅橙损失框和底部分布示意；训练流程与公式已按当前仓库实现重写。v2 将全部文本与公式统一为原图使用的 **Comic Sans MS**，并将 Strong、Weak 和分类器改为带可编辑节点与连线的小网络；系数模块增加阶梯曲线，Inception 增加特征层图标。

## 文件

- **`classifier_guidance_method_v2.pptx`**：两页均为可编辑原生对象，内嵌 Comic Sans MS 常规／粗体的完整字符集。第 1 页是 SiT 弱头与系数联合训练；第 2 页是 JiT 仅训练系数。
- **`method_joint_v2.svg` / `.pdf` / `.png`**：联合训练主图，分别用于矢量编辑、论文插图和预览。
- **`method_schedule_v2.svg` / `.pdf` / `.png`**：仅学习系数版本。
- `CAPTION.md`：英文图注与符号定义，并附中文说明。
- `REVIEW.md`、`review_checks.json`：内容审查、视觉修订和结构检查记录。
- `build_figure.py`、`drawing.py`、`embedded_fonts.py`、`*.scene.json`：生成源码和每个对象的参数。
- `fonts/`：Comic Sans MS 常规／粗体、原始安装包及 EULA。PowerPoint 若发生字体替换，安装 `Comic.TTF`、`Comicbd.TTF` 后重新打开。

不带 `_v2` 的文件与上述最新文件内容相同，用于保留脚本与已有链接的兼容性。网络节点数、图层数量和小曲线形状用于示意模块类型，不代表真实网络的精确层数、宽度或实测系数。

PowerPoint 中可选中组内对象或取消组合，修改文本、公式上下标、箭头顶点、模块颜色及单个散点。只有噪声纹理是可替换图片，流程图和散点没有被压成整张图片。箭头是自由矢量路径，移动模块后需同步调整。

## 这次改图的内容

1. 原图的 Generator 改为包含冻结强预测器、弱预测器和有符号系数的完整采样过程，公式为 `S + aα(t)(S − Wθ)`。
2. 判别器读取最终 RGB 的冻结 Inception 特征，不读取中间噪声状态或 latent。
3. 左栏冻结分类器，通过完整离散采样过程反传；右栏固定采样器并截断生成分支，只更新分类器。
4. 用实际 non-saturating logistic GAN 目标替换 FD 目标，保留真实特征上的 R1 正则。
5. 联合训练主图包含来自独立真实数据插值 probe 的 gap-energy 软尺度锚点。仅学系数版不含该项，弱头也改为冻结。
6. 右下角保留相同的真实／生成散点，只调整分类边界。左下角固定真实分布，示意生成终点分布的变化。两处均为说明性图形，不是实验结果。

图的左右顺序用于解释两类更新。实际代码每轮先更新 D，再用更新后的 D 更新引导参数。

## 内容依据

- [`classifier_guidance/README.md`](../../../classifier_guidance/README.md)：当前训练范围和冻结对象。
- [`sit_joint.py`](../../../classifier_guidance/sit_joint.py)：联合场、gap-energy probe 与 D→joint 更新。
- [`jit_schedule.py`](../../../classifier_guidance/jit_schedule.py)：固定 JiT 强弱网络、clean prediction 混合和完整求解器。
- [`schedules.py`](../../../classifier_guidance/schedules.py)：无正数约束的逐区间系数。
- [`training.py`](../../../classifier_guidance/training.py)、[`training_accumulation.py`](../../../classifier_guidance/training_accumulation.py)：真假分类反馈和真实特征 R1。
- [`binary_critic.py`](../../../experiments/adversarial_weak_training_20260915/binary_critic.py)：条件 logit 与 softplus 损失。

视觉参考：[AdvFD, Figure 3](https://arxiv.org/html/2608.11205v1#S4.F3)。仅沿用视觉表达方式；本图不再表示其 Fréchet 距离或表征对抗训练方法。

## 重建

```bash
python -m pip install python-pptx cairosvg Pillow fonttools
python build_figure.py
```

在本仓库原位置运行会同时更新源文件 SHA-256 和语义检查结果。独立使用文件包仍可重建图形；无法找到的仓库源文件哈希会记为 `null`。图形本身不读取训练数据或 checkpoint。
