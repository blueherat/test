# AdvFD Figure 3 可编辑底稿

来源：Mingju Gao, Jingkai Zhou, Kun Gai, Changqian Yu, Hao Tang,
*AdvFD: Boosting Visual Generation via Adversarial Fréchet Distance Loss*，
[arXiv:2608.11205v1](https://arxiv.org/abs/2608.11205v1)，2026-08-11，Figure 3。
原始矢量图来自该版本的 [arXiv 源码包](https://arxiv.org/src/2608.11205v1) 中的 `images/method_new.pdf`。

## 打开哪个文件

- **`advfd_figure3_editable.pptx`**：推荐作为修改入口。第 1 页是可编辑重绘，第 2 页是原图对照。第 1 页包含 28 个原生文字／公式框、80 个矢量路径、654 个原生图形和 2 个可替换的噪声纹理图片，按 7 个组整理。
- **`advfd_figure3_editable.svg`**：矢量编辑版本，可用 Inkscape / Illustrator 打开；文字仍为文字，散点仍为独立椭圆。
- `advfd_figure3_preview.png`：SVG 的高清预览。
- `advfd_figure3_editable.pdf`：矢量导出版，方便预览和插入 LaTeX。
- `reference/advfd_figure3_original.pdf`：论文源码中的原始矢量图。
- `scene.json`、`build_figure.py`：对象参数和生成脚本。

## 编辑方法

1. 打开 PowerPoint，使用第 1 页。在组内再次单击对象即可选中文字或形状；也可右键取消组合。
2. 文字、公式上下标可以直接编辑。模块与文字分开，因此移动完整模块时同时选择两者。
3. 箭头是原生矢量路径，可移动、改色或编辑顶点；移动模块后需相应调整连线。
4. 底部散点分布按 G-before、G-after、D-before、D-after 分组。取消组合后可改动每个点和轮廓。
5. 若字体发生替换，安装 `fonts/` 中的 Comic Neue 和 DejaVu Serif 字体后重新打开。字体许可一并保留。

## 复刻范围

保留了原图的模块位置、框体和连线几何、配色、两栏结构、原始公式与含义。
为方便修改，文字改用开放字体 Comic Neue / DejaVu Serif，火焰、雪花和底部分布示意重新绘制。
散点为说明性重绘，**并非论文实验数据或逐点提取结果**。因此这是一份布局和内容对应的可编辑复刻，不是像素级一致的原图。
除噪声纹理外，第 1 页没有把主要结构或分布示意压成图片。

当前内容仍然表示 **AdvFD 的方法**，不是本项目的最终方法图。来源信息保存在本说明、PPT 备注和 SVG 描述中。

## 改成本项目方法时的对应关系

根据当前 `classifier_guidance/README.md` 与训练实现，后续应实质性替换以下内容：

| AdvFD 底稿 | 本项目需要表达的内容 |
|---|---|
| 更新生成器 Gθ | 冻结强生成模型；学习弱头、引导系数或两者，具体随实验设置变化 |
| Static / Adverse representation 两条 FD 分支 | 冻结图像特征网络与可训练真假分类器 |
| 最小化／最大化 Fréchet discrepancy | 最终生成图像上的二分类对抗目标 |
| 一步生成器到 Fake Batch | 完整引导采样轨迹到最终图像，梯度沿完整轨迹回传 |
| 分布拉近／推远示意 | 对应弱头／系数更新与分类器更新的示意；需要重新确定坐标含义 |

原图 D-step 中的静态分支与汇合线按原样保留；原论文 D-step 优化目标本身仅包含对抗 FD 项。

## 重新生成与检查

```bash
python -m pip install python-pptx pymupdf cairosvg Pillow
python build_figure.py
```

已目视检查 SVG 预览，检查 PPTX 可重新读取且包含两页、原生文字和独立图形，并检查文件包结构。
当前环境没有 PowerPoint，因此未直接在 PowerPoint 内渲染；跨软件的字体度量可能略有差异。
